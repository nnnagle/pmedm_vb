"""The download cache: reuse, and refusing the network when offline."""

import pytest

from pmedm_vb.data import cache


def test_a_cached_file_is_returned_without_a_download(tmp_path, monkeypatch):
    monkeypatch.setenv(cache.OFFLINE_ENV, "1")
    dest = tmp_path / "file.csv"
    dest.write_text("x")
    assert cache.fetch("https://example.invalid/file.csv", dest) == dest


def test_offline_mode_refuses_a_missing_file_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv(cache.OFFLINE_ENV, "1")
    dest = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="missing.csv.*prefetch"):
        cache.fetch("https://example.invalid/missing.csv", dest)
    assert not dest.exists()
