import pytest

from smog_service import registry


def test_artifact_hash_is_verified(monkeypatch, tmp_path):
    fake = tmp_path / "forecaster.joblib"
    fake.write_bytes(b"not the real model")
    with pytest.raises(RuntimeError, match="does not match"):
        registry.load(fake)


def test_fetch_rejects_tampered_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "CACHE_DIR", tmp_path)
    cached = tmp_path / registry.MODEL_COMMIT[:12] / "models" / "forecaster.joblib"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        registry.fetch("models/forecaster.joblib", registry.MODEL_SHA256)
    assert not cached.exists()  # the bad file is removed so the next start re-downloads


def test_raw_url_is_pinned_to_a_commit():
    url = registry.raw_url("models/forecaster.joblib")
    assert registry.MODEL_COMMIT in url and "/main/" not in url
