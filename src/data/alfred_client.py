"""ALFRED/FRED vintage downloader.

Implements docs/01_data_spec.md §1-2 and milestone M0 task 1.

Rules:
- R4: raw responses cached immutably under ``data/raw/``; a second run must
  not hit the network. Manifest records endpoint, date and content hash.
- R5: public sources only. Requires ``FRED_API_KEY`` env var (free).

Cache layout (see docs/00_master_plan.md M0 acceptance)
--------------------------------------------------------
``data/raw/fred/vintagedates/{series_id}.json``
    ``{"series_id", "vintage_dates": [iso date, ...], "fetched_at"}``
``data/raw/fred/observations/{series_id}/{vintage_date}.json``
    ``{"series_id", "vintage_date", "observations": [{"date", "value"}, ...],
    "fetched_at"}`` — a single ALFRED "as known on vintage_date" snapshot.
``data/raw/manifest.json``
    append-only list of ``{endpoint, params, cache_path, request_sha1,
    content_sha256, fetched_at}`` entries, one per actual network fetch
    (cache hits are not re-logged). ``content_sha256`` hashes the bytes
    written, so the cache can be verified; ``request_sha1`` only identifies
    the request. NOTE: entries written before the M0 security review carry
    only a legacy ``sha1`` (the request hash) and no content hash — see.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"
CACHE_DIR = Path("data/raw/fred")
MANIFEST_PATH = Path("data/raw/manifest.json")
RATE_LIMIT_SECONDS = 0.5
MAX_RETRIES = 5

# FRED series IDs are alphanumeric/underscore. Anything else is rejected before
# it can reach a filesystem path (a config-supplied id like "../../x" or an
# absolute path would otherwise escape CACHE_DIR).
_SERIES_ID_RE = re.compile(r"[A-Za-z0-9_]{1,64}\Z")
_APIKEY_RE = re.compile(r"(api_key=)[^&\s]+")


def _redact(value: object) -> str:
    """Strip the API key out of anything destined for a log or an exception.

    ``requests`` embeds the full request URL — query string included — in the
    string form of transport errors and of ``raise_for_status``, so exception
    text must never be interpolated verbatim.
    """
    return _APIKEY_RE.sub(r"\1***", str(value))


def _check_series_id(series_id: str) -> str:
    if not isinstance(series_id, str) or not _SERIES_ID_RE.match(series_id):
        raise ValueError(f"invalid FRED series id: {series_id!r}")
    return series_id


def _cache_key(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _content_hash(path: Path) -> str:
    """SHA-256 of the bytes actually written to the cache (R4 integrity)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("Set FRED_API_KEY (free at fred.stlouisfed.org)")
    return key


def _append_manifest(entry: dict) -> None:
    """Append one provenance record, atomically.

    Written via a temp file + ``os.replace`` so an interrupt mid-write cannot
    truncate the manifest (it is the only provenance artifact for a 3 GB
    cache). NOTE: this is still a full read-modify-write and therefore O(n^2)
    over a download; see for the planned move to JSONL.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if MANIFEST_PATH.exists():
        entries = json.loads(MANIFEST_PATH.read_text())
    entries.append(entry)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2, sort_keys=True))
    os.replace(tmp, MANIFEST_PATH)


def _get(endpoint: str, params: dict) -> dict:
    """GET a FRED endpoint with retries/backoff. Fails loudly on 4xx."""
    url = f"{FRED_BASE}/{endpoint}"
    query = {**params, "api_key": _api_key(), "file_type": "json"}
    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=query, timeout=30, allow_redirects=False)
        except requests.exceptions.RequestException as exc:
            # _redact: transport errors stringify the full URL, key included.
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"FRED {endpoint} failed after {MAX_RETRIES} retries: "
                    f"{_redact(exc)}"
                ) from None
            logger.warning(
                "FRED %s request error %s (attempt %d/%d); backing off %.1fs",
                endpoint,
                _redact(exc),
                attempt,
                MAX_RETRIES,
                backoff,
            )
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.is_redirect or resp.is_permanent_redirect:
            # Never follow a redirect: the key rides in the query string and
            # would be handed to whatever host we were redirected to.
            raise RuntimeError(
                f"FRED {endpoint} unexpectedly redirected ({resp.status_code}); refusing to follow"
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == MAX_RETRIES:
                # Not raise_for_status(): its message embeds the full URL.
                raise RuntimeError(
                    f"FRED {endpoint} failed with {resp.status_code} after "
                    f"{MAX_RETRIES} retries"
                )
            logger.warning(
                "FRED %s returned %d (attempt %d/%d); backing off %.1fs",
                endpoint,
                resp.status_code,
                attempt,
                MAX_RETRIES,
                backoff,
            )
            time.sleep(backoff)
            backoff *= 2
            continue
        # 4xx other than 429: do not retry, do not fabricate data (R5).
        raise RuntimeError(
            f"FRED {endpoint} failed with {resp.status_code}: {resp.text[:500]}"
        )
    raise RuntimeError(f"FRED {endpoint} failed after {MAX_RETRIES} retries")


def get_vintage_dates(series_id: str) -> list[date]:
    """Return all vintage dates for a series (ALFRED), cached under CACHE_DIR."""
    _check_series_id(series_id)
    cache_path = CACHE_DIR / "vintagedates" / f"{series_id}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
    else:
        payload = _get("series/vintagedates", {"series_id": series_id})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "series_id": series_id,
            "vintage_dates": payload.get("vintage_dates", []),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        _append_manifest(
            {
                "endpoint": "series/vintagedates",
                "params": {"series_id": series_id},
                "cache_path": str(cache_path),
                "request_sha1": _cache_key("series/vintagedates", series_id),
                "content_sha256": _content_hash(cache_path),
                "fetched_at": record["fetched_at"],
            }
        )
        time.sleep(RATE_LIMIT_SECONDS)
        payload = record
    dates = payload["vintage_dates"]
    return [date.fromisoformat(d) for d in dates]


def get_series_at_vintage(series_id: str, vintage_date: date) -> pd.Series:
    """Return observations of ``series_id`` as known on ``vintage_date``."""
    _check_series_id(series_id)
    vd_str = vintage_date.isoformat()
    cache_path = CACHE_DIR / "observations" / series_id / f"{vd_str}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
    else:
        raw = _get(
            "series/observations",
            {
                "series_id": series_id,
                "vintage_dates": vd_str,
            },
        )
        observations = [
            {"date": o["date"], "value": o["value"]}
            for o in raw.get("observations", [])
        ]
        payload = {
            "series_id": series_id,
            "vintage_date": vd_str,
            "observations": observations,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        _append_manifest(
            {
                "endpoint": "series/observations",
                "params": {"series_id": series_id, "vintage_dates": vd_str},
                "cache_path": str(cache_path),
                "request_sha1": _cache_key("series/observations", series_id, vd_str),
                "content_sha256": _content_hash(cache_path),
                "fetched_at": payload["fetched_at"],
            }
        )
        time.sleep(RATE_LIMIT_SECONDS)

    obs = payload["observations"]
    if not obs:
        return pd.Series(dtype=float, name=vintage_date.isoformat())
    idx = pd.to_datetime([o["date"] for o in obs])
    vals = [
        float(o["value"]) if o["value"] not in (".", "", None) else float("nan")
        for o in obs
    ]
    return pd.Series(vals, index=idx, name=vintage_date.isoformat()).sort_index()


PANELS = ("baseline", "extended", "all")


def panel_series_ids(config: dict, panel: str = "all") -> list[str]:
    """Series ids for a panel tier, EXCLUDING the target.

    ``baseline`` keeps only series whose real ALFRED vintage history spans the
    whole evaluation window; ``extended`` adds the late-starting ones, which
    are robustness-only. See.
    """
    if panel not in PANELS:
        raise ValueError(f"panel must be one of {PANELS}, got {panel!r}")
    entries = config["series"]
    if panel == "baseline":
        entries = [e for e in entries if e.get("panel", "baseline") == "baseline"]
    return [e["fred_id"] for e in entries]


def _panel_series_ids(config: dict, panel: str = "all") -> list[str]:
    """Panel series plus the target (what actually needs downloading)."""
    series_ids = panel_series_ids(config, panel)
    target_id = config.get("target", {}).get("fred_id")
    if target_id and target_id not in series_ids:
        series_ids = [target_id] + series_ids
    return series_ids


def backfill_manifest_hashes(manifest_path: Path | None = None) -> dict:
    """Add ``content_sha256`` to manifest entries written before the fix.

    Entries recorded before the M0 security review
    carry only a request hash, so the cache they describe cannot be verified.
    This walks those entries, hashes the bytes actually on disk, and rewrites
    the manifest atomically. Entries whose cache file is missing are reported
    rather than dropped -- a manifest that quietly forgets a missing file is
    worse than one that admits it.
    """
    path = manifest_path or MANIFEST_PATH
    entries = json.loads(path.read_text())
    filled = missing = already = 0
    for entry in entries:
        if entry.get("content_sha256"):
            already += 1
            continue
        cache_path = Path(entry.get("cache_path", ""))
        if not cache_path.exists():
            entry["content_missing"] = True
            missing += 1
            continue
        entry["content_sha256"] = _content_hash(cache_path)
        entry["content_hash_backfilled"] = True
        filled += 1

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return {
        "entries": len(entries),
        "already_hashed": already,
        "backfilled": filled,
        "cache_file_missing": missing,
    }


def verify_cache(manifest_path: Path | None = None, sample: int | None = None) -> dict:
    """Re-hash cached files and compare against the manifest (R4).

    ``sample`` checks a random subset, which is what makes this usable as a
    routine check on a 3 GB cache rather than something nobody ever runs.
    """
    import random as _random

    path = manifest_path or MANIFEST_PATH
    entries = [e for e in json.loads(path.read_text()) if e.get("content_sha256")]
    if sample and sample < len(entries):
        entries = _random.Random(42).sample(entries, sample)

    ok = mismatched = missing = 0
    bad: list[str] = []
    for entry in entries:
        cache_path = Path(entry["cache_path"])
        if not cache_path.exists():
            missing += 1
            bad.append(f"missing: {cache_path}")
            continue
        if _content_hash(cache_path) == entry["content_sha256"]:
            ok += 1
        else:
            mismatched += 1
            bad.append(f"hash mismatch: {cache_path}")
    return {
        "checked": len(entries),
        "ok": ok,
        "mismatched": mismatched,
        "missing": missing,
        "problems": bad[:20],
    }


def download_panel(config: dict, panel: str = "all") -> None:
    """Download vintages for every series in the given panel, plus the target.

    Series are fetched BASELINE-FIRST. The baseline tier carries the study's
    headline results, while the extended tier is robustness-only and is used
    in pseudo-real-time anyway -- and it is also
    where the cost is concentrated: the daily series carry their full history
    in every snapshot (~822 KB per NASDAQCOM vintage vs ~18 KB for GDPC1), so
    the extended tier is ~90% of the bytes. Ordering this way means an
    interrupted download still leaves the part the study depends on complete.

    Per-series failures are collected rather than aborting the run, so one bad
    series id cannot strand the rest; the run still FAILS at the end with a
    summary, because silently returning a partial panel is what R5 forbids.
    """
    baseline = set(_panel_series_ids(config, "baseline"))
    ordered = sorted(
        _panel_series_ids(config, panel), key=lambda s: (s not in baseline,)
    )

    failures: dict[str, str] = {}
    for series_id in ordered:
        try:
            vintage_dates = get_vintage_dates(series_id)
            logger.info("%s: %d vintage dates", series_id, len(vintage_dates))
            for vd in vintage_dates:
                get_series_at_vintage(series_id, vd)
        except Exception as exc:  # noqa: BLE001 - reported, then re-raised below
            logger.error("%s: download failed: %s", series_id, _redact(exc))
            failures[series_id] = str(_redact(exc))

    if failures:
        summary = "; ".join(f"{k}: {v}" for k, v in failures.items())
        raise RuntimeError(f"{len(failures)} series failed to download - {summary}")


def coverage_table(config: dict, strict: bool = True) -> pd.DataFrame:
    """Per-series coverage summary (M0 acceptance, docs/00_master_plan.md).

    ``n_vintages_expected`` comes from the authoritative ALFRED vintage-date
    listing, not from the cache directory, so an interrupted download shows up
    as an incomplete row instead of being reported as a complete series.
    With ``strict`` (the default) any incomplete series raises.
    """
    rows = []
    incomplete = []
    for series_id in _panel_series_ids(config):
        obs_dir = CACHE_DIR / "observations" / series_id
        vintage_files = sorted(obs_dir.glob("*.json")) if obs_dir.exists() else []
        n_cached = len(vintage_files)

        # Ground truth: what ALFRED says exists (cached, so no network call).
        vd_path = CACHE_DIR / "vintagedates" / f"{series_id}.json"
        n_expected = None
        if vd_path.exists():
            n_expected = len(json.loads(vd_path.read_text())["vintage_dates"])

        complete = n_expected is not None and n_cached == n_expected
        if not complete:
            incomplete.append(f"{series_id} ({n_cached}/{n_expected})")

        first_vintage = vintage_files[0].stem if vintage_files else None
        last_vintage = vintage_files[-1].stem if vintage_files else None
        first_obs, last_obs = None, None
        if vintage_files:
            last_payload = json.loads(vintage_files[-1].read_text())
            obs_dates = [o["date"] for o in last_payload["observations"]]
            if obs_dates:
                first_obs, last_obs = min(obs_dates), max(obs_dates)
        rows.append(
            {
                "fred_id": series_id,
                "n_vintages": n_cached,
                "n_vintages_expected": n_expected,
                "complete": complete,
                "first_vintage_date": first_vintage,
                "last_vintage_date": last_vintage,
                "first_obs_date": first_obs,
                "last_obs_date": last_obs,
            }
        )

    if strict and incomplete:
        raise RuntimeError(
            "coverage is incomplete for "
            f"{len(incomplete)} series: {', '.join(incomplete)}. "
            "Re-run `make data` (cached vintages are skipped) or pass "
            "strict=False to emit the partial table deliberately."
        )
    return pd.DataFrame(rows)
