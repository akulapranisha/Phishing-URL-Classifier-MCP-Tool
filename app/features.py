"""Shared lexical and host-based URL feature extraction for training and serving."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qs, urlparse

# Ordered feature names — must remain stable across train and serve.
FEATURE_NAMES: list[str] = [
    "url_length",
    "host_length",
    "path_depth",
    "subdomain_count",
    "digit_ratio",
    "special_char_count",
    "has_ip_as_host",
    "has_at_symbol",
    "hyphen_count",
    "url_entropy",
    "tld_length",
    "is_https",
    "suspicious_keyword_count",
    "dot_count",
    "slash_count",
    "query_param_count",
    "has_port",
    "double_slash_in_path",
    "percent_encoding_count",
    "uppercase_ratio",
    "longest_token_length",
    "has_redirect_keyword",
    "host_digit_count",
]

SUSPICIOUS_KEYWORDS: frozenset[str] = frozenset(
    {
        "login",
        "signin",
        "verify",
        "update",
        "secure",
        "account",
        "banking",
        "password",
        "confirm",
        "wallet",
        "paypal",
        "suspend",
        "unlock",
        "credential",
        "validation",
    }
)

REDIRECT_KEYWORDS: frozenset[str] = frozenset(
    {"redirect", "redir", "goto", "url=", "next=", "return=", "continue="}
)

IPV4_PATTERN = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"
)
SPECIAL_CHARS = set("!@#$%^&*()=+[]{}|;:'\",<>?`~")


def extract_features(url: str) -> dict[str, float | int]:
    """Extract 23 lexical and host-based features from a URL string.

    Features (exactly 23):
        1.  url_length              — total character count of the URL
        2.  host_length             — character count of the hostname
        3.  path_depth              — number of non-empty path segments
        4.  subdomain_count         — number of dot-separated labels before the registrable domain
        5.  digit_ratio             — fraction of characters that are digits
        6.  special_char_count      — count of non-alphanumeric, non-space punctuation
        7.  has_ip_as_host          — 1 if hostname is an IPv4 literal, else 0
        8.  has_at_symbol           — 1 if '@' appears in the URL (credential-obfuscation signal)
        9.  hyphen_count            — total hyphen characters in the URL
        10. url_entropy             — Shannon entropy of URL characters (bits)
        11. tld_length              — length of the top-level domain label
        12. is_https                — 1 if scheme is https, else 0
        13. suspicious_keyword_count — count of phishing-related keywords in URL (lowercased)
        14. dot_count               — total '.' characters in the URL
        15. slash_count             — total '/' characters in the URL
        16. query_param_count       — number of distinct query parameters
        17. has_port                — 1 if an explicit non-default port is present
        18. double_slash_in_path    — count of '//' sequences after the scheme (path obfuscation)
        19. percent_encoding_count  — count of '%XX' percent-encoded byte sequences
        20. uppercase_ratio         — fraction of alphabetic characters that are uppercase
        21. longest_token_length    — length of the longest alphanumeric token
        22. has_redirect_keyword    — 1 if redirect-style keywords appear, else 0
        23. host_digit_count        — digit characters in the hostname

    Args:
        url: Raw URL string (with or without scheme).

    Returns:
        Dictionary mapping feature name to numeric value.
    """
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full_lower = normalized.lower()

    url_len = len(normalized)
    digits = sum(ch.isdigit() for ch in normalized)
    alpha = [ch for ch in normalized if ch.isalpha()]
    uppercase = sum(ch.isupper() for ch in alpha)

    tokens = re.findall(r"[a-zA-Z0-9]+", normalized)
    longest_token = max((len(t) for t in tokens), default=0)

    path_segments = [seg for seg in path.split("/") if seg]
    subdomain_count = _count_subdomains(host)
    tld = _extract_tld(host)

    suspicious_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full_lower)
    redirect_hit = int(any(kw in full_lower for kw in REDIRECT_KEYWORDS))

    path_and_query = f"{path}?{query}" if query else path
    double_slash_in_path = path_and_query.count("//")

    default_ports = {("http", 80), ("https", 443)}
    has_port = int(
        parsed.port is not None
        and (parsed.scheme or "http", parsed.port) not in default_ports
    )

    return {
        "url_length": float(url_len),
        "host_length": float(len(host)),
        "path_depth": float(len(path_segments)),
        "subdomain_count": float(subdomain_count),
        "digit_ratio": float(digits / url_len) if url_len else 0.0,
        "special_char_count": float(sum(ch in SPECIAL_CHARS for ch in normalized)),
        "has_ip_as_host": float(bool(host and IPV4_PATTERN.match(host))),
        "has_at_symbol": float("@" in normalized),
        "hyphen_count": float(normalized.count("-")),
        "url_entropy": float(_shannon_entropy(normalized)),
        "tld_length": float(len(tld)),
        "is_https": float(parsed.scheme == "https"),
        "suspicious_keyword_count": float(suspicious_hits),
        "dot_count": float(normalized.count(".")),
        "slash_count": float(normalized.count("/")),
        "query_param_count": float(len(parse_qs(query))),
        "has_port": float(has_port),
        "double_slash_in_path": float(double_slash_in_path),
        "percent_encoding_count": float(len(re.findall(r"%[0-9a-fA-F]{2}", normalized))),
        "uppercase_ratio": float(uppercase / len(alpha)) if alpha else 0.0,
        "longest_token_length": float(longest_token),
        "has_redirect_keyword": float(redirect_hit),
        "host_digit_count": float(sum(ch.isdigit() for ch in host)),
    }


def features_to_vector(features: dict[str, Any]) -> list[float]:
    """Convert a feature dict to an ordered vector matching FEATURE_NAMES."""
    return [float(features[name]) for name in FEATURE_NAMES]


def _normalize_url(url: str) -> str:
    """Ensure URL has a scheme for consistent parsing."""
    url = url.strip()
    if not url:
        return "http://"
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return f"http://{url}"
    return url


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy in bits."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


def _count_subdomains(host: str) -> int:
    """Count subdomain labels (approximation without public-suffix list)."""
    if not host or IPV4_PATTERN.match(host):
        return 0
    parts = host.split(".")
    if len(parts) <= 2:
        return max(0, len(parts) - 1)
    return len(parts) - 2


def _extract_tld(host: str) -> str:
    """Return the rightmost domain label as a TLD approximation."""
    if not host or IPV4_PATTERN.match(host):
        return ""
    parts = host.split(".")
    return parts[-1] if parts else ""
