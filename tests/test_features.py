"""Unit tests for shared feature extraction."""

from __future__ import annotations

import pytest

from app.features import FEATURE_NAMES, extract_features


@pytest.mark.parametrize(
    "url,expected_keys",
    [
        ("https://www.google.com", set(FEATURE_NAMES)),
        ("http://192.168.1.1/login", set(FEATURE_NAMES)),
        ("paypal-secure-login.com/verify", set(FEATURE_NAMES)),
    ],
)
def test_extract_features_returns_all_23_features(url: str, expected_keys: set[str]) -> None:
    features = extract_features(url)
    assert set(features.keys()) == expected_keys
    assert len(features) == 23


def test_benign_url_low_suspicious_signals() -> None:
    features = extract_features("https://www.python.org/downloads/")
    assert features["is_https"] == 1.0
    assert features["has_ip_as_host"] == 0.0
    assert features["has_at_symbol"] == 0.0
    assert features["suspicious_keyword_count"] == 0.0


def test_phishing_url_high_suspicious_signals() -> None:
    features = extract_features("http://paypa1-secure-login.com/signin?account=suspended")
    assert features["has_ip_as_host"] == 0.0
    assert features["suspicious_keyword_count"] >= 2.0
    assert features["hyphen_count"] >= 2.0
    assert features["is_https"] == 0.0


def test_ip_host_detected() -> None:
    features = extract_features("http://192.168.0.1/paypal-login")
    assert features["has_ip_as_host"] == 1.0


def test_at_symbol_detected() -> None:
    features = extract_features("http://user:pass@evil-phish.com/login")
    assert features["has_at_symbol"] == 1.0


def test_feature_values_are_numeric() -> None:
    features = extract_features("https://example.com/path")
    for name, value in features.items():
        assert isinstance(value, (int, float)), f"{name} is not numeric"
