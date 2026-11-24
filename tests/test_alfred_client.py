"""Security and integrity tests for the ALFRED client (R4/R5).

Covers the issues found in the M0 security review:
the API key must never reach a log or an exception, series ids must not be
able to escape the cache directory, and the manifest must record a real
content hash.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
import requests

from src.data import alfred_client as ac


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.is_redirect = status_code in (301, 302, 303, 307, 308)
        self.is_permanent_redirect = status_code in (301, 308)

    def json(self) -> dict:
        return self._payload


SECRET = "0123456789abcdef0123456789abcdef"


def test_redact_strips_api_key():
    leaky = (
        "HTTPSConnectionPool(host='api.stlouisfed.org', port=443): Max retries "
        f"exceeded with url: /fred/series/observations?series_id=GDPC1&api_key={SECRET}"
    )
    assert SECRET not in ac._redact(leaky)
    assert "api_key=***" in ac._redact(leaky)


def test_transport_error_never_leaks_key(monkeypatch, caplog):
    """A ConnectionError stringifies the full URL; it must be redacted (F1)."""
    monkeypatch.setenv("FRED_API_KEY", SECRET)
    monkeypatch.setattr(ac, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(ac.time, "sleep", lambda *_: None)

    def _boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError(
            "Max retries exceeded with url: /fred/series/observations?"
            f"series_id=GDPC1&api_key={SECRET}"
        )

    monkeypatch.setattr(ac.requests, "get", _boom)

    with caplog.at_level("WARNING"), pytest.raises(RuntimeError) as excinfo:
        ac._get("series/observations", {"series_id": "GDPC1"})

    assert SECRET not in str(excinfo.value)
    # ...and not in the chained context either (we raise `from None`).
    assert SECRET not in repr(excinfo.value.__cause__)
    for record in caplog.records:
        assert SECRET not in record.getMessage()


def test_http_error_never_leaks_key(monkeypatch, caplog):
    """raise_for_status() embeds the URL; we must not use it (F1)."""
    monkeypatch.setenv("FRED_API_KEY", SECRET)
    monkeypatch.setattr(ac, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(ac.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ac.requests, "get", lambda *a, **k: _FakeResponse(500))

    with caplog.at_level("WARNING"), pytest.raises(RuntimeError) as excinfo:
        ac._get("series/observations", {"series_id": "GDPC1"})

    assert SECRET not in str(excinfo.value)
    assert "500" in str(excinfo.value)
    for record in caplog.records:
        assert SECRET not in record.getMessage()


def test_redirects_are_refused_not_followed(monkeypatch):
    """The key rides in the query string, so a redirect must never be followed (F6)."""
    monkeypatch.setenv("FRED_API_KEY", SECRET)
    monkeypatch.setattr(ac.requests, "get", lambda *a, **k: _FakeResponse(302))

    with pytest.raises(RuntimeError, match="redirect"):
        ac._get("series/observations", {"series_id": "GDPC1"})


@pytest.mark.parametrize(
    "bad_id",
    [
        "../../../../tmp/pwned",
        "/etc/cron.d/evil",
        "GDPC1/../../escape",
        "with space",
        "",
        "x" * 65,
    ],
)
def test_series_id_cannot_escape_cache_dir(bad_id, tmp_path, monkeypatch):
    """Config-supplied ids are interpolated into paths; they must be validated (F3)."""
    monkeypatch.setattr(ac, "CACHE_DIR", tmp_path)
    with pytest.raises(ValueError, match="invalid FRED series id"):
        ac.get_vintage_dates(bad_id)
    with pytest.raises(ValueError, match="invalid FRED series id"):
        ac.get_series_at_vintage(bad_id, date(2020, 1, 1))


def test_valid_series_ids_accepted():
    for good in ("GDPC1", "PAYEMS", "CES0600000007", "DGORDER", "A_B_1"):
        assert ac._check_series_id(good) == good


def test_manifest_records_real_content_hash(tmp_path, monkeypatch):
    """`content_sha256` must hash the cached bytes, not the request params (F2)."""
    import hashlib

    monkeypatch.setenv("FRED_API_KEY", SECRET)
    monkeypatch.setattr(ac, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ac, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(ac, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(
        ac.requests,
        "get",
        lambda *a, **k: _FakeResponse(
            200, {"observations": [{"date": "2020-01-01", "value": "1.5"}]}
        ),
    )

    ac.get_series_at_vintage("GDPC1", date(2020, 3, 1))

    entries = json.loads((tmp_path / "manifest.json").read_text())
    assert len(entries) == 1
    cache_path = tmp_path / "observations" / "GDPC1" / "2020-03-01.json"
    expected = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    assert entries[0]["content_sha256"] == expected

    # And it must actually change if the cached bytes are tampered with.
    cache_path.write_text(json.dumps({"observations": [], "tampered": True}))
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() != expected


def test_manifest_append_is_atomic(tmp_path, monkeypatch):
    """A crash mid-write must not truncate the only provenance artifact (F4)."""
    monkeypatch.setattr(ac, "MANIFEST_PATH", tmp_path / "manifest.json")
    ac._append_manifest({"a": 1})
    ac._append_manifest({"a": 2})
    entries = json.loads((tmp_path / "manifest.json").read_text())
    assert entries == [{"a": 1}, {"a": 2}]
    # No temp file left behind.
    assert not list(tmp_path.glob("*.tmp"))


def test_panel_tiers_select_the_right_series():
    """The two paper specifications must be selectable from config."""
    import yaml

    with open("configs/data.yaml") as fh:
        cfg = yaml.safe_load(fh)
    baseline = ac.panel_series_ids(cfg, "baseline")
    everything = ac.panel_series_ids(cfg, "all")

    # Baseline is the constant-composition panel used for headline results.
    assert set(baseline) < set(everything)
    late = set(everything) - set(baseline)
    assert late == {"ICSA", "DCOILWTICO", "BAA10Y", "T10YFF", "DEXUSEU", "NASDAQCOM"}

    # docs/01_data_spec.md §2 requires at least 20 indicators; the baseline
    # panel must satisfy it on its own, without the robustness-only block.
    assert len(baseline) >= 20, f"baseline panel has only {len(baseline)} indicators"

    # Every baseline series must actually have vintages predating the sample.
    start = pd.Timestamp(cfg["sample"]["nowcast_start"])
    for entry in cfg["series"]:
        if entry.get("panel", "baseline") != "baseline":
            continue
        assert pd.Timestamp(entry["vintage_start"]) <= start, (
            f"{entry['fred_id']} is tagged baseline but its vintages start "
            f"{entry['vintage_start']}, after the sample opens"
        )

    with pytest.raises(ValueError, match="panel must be one of"):
        ac.panel_series_ids(cfg, "nonsense")


def test_target_is_not_in_the_panel_series_list():
    """The target must not be a regressor by default (self-leak, R1)."""
    import yaml

    with open("configs/data.yaml") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["target"]["fred_id"] not in ac.panel_series_ids(cfg, "all")
