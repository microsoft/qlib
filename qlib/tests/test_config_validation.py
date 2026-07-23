import pytest

from qlib.config import Config


def test_missing_provider_uri_raises():
    default_conf = {
        "provider_uri": None,
        "region": "us",
    }

    cfg = Config(default_conf)

    with pytest.raises(ValueError) as exc:
        cfg.validate()

    assert "provider_uri must be set" in str(exc.value)
