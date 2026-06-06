"""Generate a small labeled sample dataset for out-of-the-box training."""

from __future__ import annotations

import csv
from pathlib import Path

BENIGN = [
    "https://www.google.com/search?q=weather",
    "https://github.com/python/cpython",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://stackoverflow.com/questions/ask",
    "https://www.python.org/downloads/",
    "https://docs.python.org/3/library/urllib.html",
    "https://www.nytimes.com/",
    "https://www.bbc.com/news",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.microsoft.com/en-us/",
    "https://openai.com/research",
    "https://huggingface.co/models",
    "https://www.kaggle.com/datasets",
    "https://www.reddit.com/r/python/",
    "https://news.ycombinator.com/",
    "https://www.linkedin.com/feed/",
    "https://www.apple.com/iphone/",
    "https://www.spotify.com/us/",
    "https://www.netflix.com/browse",
    "https://www.cloudflare.com/learning/",
    "https://fastapi.tiangolo.com/",
    "https://scikit-learn.org/stable/",
    "https://pandas.pydata.org/docs/",
    "https://numpy.org/doc/",
    "https://www.docker.com/products/docker-desktop",
    "https://kubernetes.io/docs/home/",
    "https://www.postgresql.org/docs/",
    "https://redis.io/docs/",
    "https://www.elastic.co/guide/",
    "https://grafana.com/docs/",
    "https://www.djangoproject.com/",
    "https://flask.palletsprojects.com/",
    "https://www.npmjs.com/package/express",
    "https://react.dev/learn",
    "https://vuejs.org/guide/",
    "https://tailwindcss.com/docs",
    "https://www.typescriptlang.org/docs/",
    "https://developer.mozilla.org/en-US/docs/Web",
    "https://www.w3.org/TR/",
    "https://www.ietf.org/standards/",
    "https://www.cnn.com/world",
    "https://www.theguardian.com/international",
    "https://www.wsj.com/",
    "https://www.bloomberg.com/markets",
    "https://www.reuters.com/world/",
    "https://www.nature.com/articles",
    "https://arxiv.org/list/cs.AI/recent",
    "https://www.coursera.org/",
    "https://www.edx.org/",
    "https://www.khanacademy.org/",
]

PHISHING = [
    "http://192.168.0.1/paypal-login/secure-verify",
    "http://paypa1-secure-login.com/signin?account=suspended",
    "https://secure-banking-update.xyz/verify/account",
    "http://apple-id-locked-verify.ru/unlock",
    "https://micros0ft-account-update.tk/password/reset",
    "http://amaz0n-security-check.club/wallet/confirm",
    "https://login-facebook-security.ga/confirm/credential",
    "http://netflix-billing-update.bid/account/validation",
    "https://chase-bank-login-secure.top/online/signin",
    "http://wellsfargo-verify-account.win/update",
    "https://paypal-secure-login-confirm.xyz/signin",
    "http://google-drive-share-verify.ru/download/file",
    "https://dropbox-security-alert.ml/login/verify",
    "http://office365-password-expire.cf/update/now",
    "https://dhl-package-hold-confirm.ga/track/shipment",
    "http://fedex-delivery-fee-pay.tk/invoice",
    "https://usps-redelivery-confirm.top/pay/fee",
    "http://irs-refund-claim-update.club/form/submit",
    "https://coinbase-wallet-secure.ga/validation/unlock",
    "http://binance-login-verify.ml/account/suspend",
    "https://metamask-wallet-sync.tk/confirm/seed",
    "http://steam-community-trade-verify.ru/gift/accept",
    "https://instagram-verify-badge.ga/account/confirm",
    "http://whatsapp-security-update.bid/login",
    "https://telegram-premium-claim.cf/activate",
    "http://linkedin-premium-invoice.top/billing/update",
    "http://user:pass@evil-phish.com/paypal/login",
    "https://secure-login-confirm-bank.xyz/redirect?url=steal",
    "http://www.paypal.com.secure-login-verify.ru/signin",
    "https://account-update-secure-confirm.ga/banking/login",
    "http://signin-appleid-locked-verify.tk/unlock-now",
    "https://microsoft-account-password-reset.ml/confirm",
    "http://amazon-prime-renewal-failed.club/update-payment",
    "https://netflix-membership-hold.ga/billing/verify",
    "http://spotify-premium-expired.cf/reactivate/account",
    "https://bankofamerica-secure-signin.top/online/login",
    "http://citibank-alert-verify.win/account/update",
    "http://capitalone-fraud-alert.tk/confirm/identity",
    "https://discover-card-locked.ga/unlock/card",
    "http://americanexpress-verify.ml/account/security",
    "https://ebay-buyer-protection-claim.ru/refund/form",
    "http://walmart-gift-card-winner.club/claim/prize",
    "https://costco-membership-renew.bid/payment/update",
    "http://target-order-hold-confirm.tk/pay/shipping",
    "https://bestbuy-order-cancel-avoid.ga/confirm/order",
    "http://samsung-account-locked-verify.cf/unlock/device",
    "https://icloud-storage-full-alert.ml/upgrade/storage",
    "http://google-photos-share-expire.top/view/album",
    "https://outlook-mailbox-quota-exceeded.ga/expand/mailbox",
    "http://yahoo-mail-security-alert.tk/verify/login",
]


def main() -> None:
    output = Path("data/urls.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    for url in BENIGN:
        rows.append((url, "benign"))
    for url in PHISHING:
        rows.append((url, "phishing"))

    # Augment with variations for more training rows
    for i in range(8):
        for url in BENIGN[:30]:
            rows.append((f"{url}?ref=sample{i}", "benign"))
        for url in PHISHING[:30]:
            rows.append((url.replace(".com", f"-{i}.com"), "phishing"))

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["url", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
