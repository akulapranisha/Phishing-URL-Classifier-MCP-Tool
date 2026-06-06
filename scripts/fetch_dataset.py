"""Download a public phishing URL dataset into data/urls.csv.

Primary source: PhiUSIIL Phishing URL Dataset (UCI ML Repository)
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Alternative (smaller): OpenML URL dataset mirrors.

Usage:
    python scripts/fetch_dataset.py
    python scripts/fetch_dataset.py --output data/urls_full.csv --max-rows 80000
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
)


def fetch_uci_dataset(output: Path, max_rows: int | None = None) -> None:
    """Download and normalize the UCI PhiUSIIL dataset."""
    print(f"Downloading from {UCI_ZIP_URL} ...", file=sys.stderr)
    response = requests.get(UCI_ZIP_URL, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV found in UCI archive")
        with zf.open(csv_names[0]) as handle:
            df = pd.read_csv(handle)

    # Normalize columns — UCI file uses 'URL' and 'label' (or similar)
    col_map = {c.lower(): c for c in df.columns}
    url_col = col_map.get("url") or col_map.get("urls")
    label_col = col_map.get("label") or col_map.get("class") or col_map.get("phishing")

    if url_col is None:
        raise RuntimeError(f"Could not find URL column in: {list(df.columns)}")

    result = pd.DataFrame()
    result["url"] = df[url_col].astype(str)

    if label_col:
        labels = df[label_col]
        if labels.dtype.kind in {"i", "u", "f"}:
            result["label"] = labels.map({0: "benign", 1: "phishing", 0.0: "benign", 1.0: "phishing"})
        else:
            result["label"] = labels.astype(str).str.lower()
    else:
        raise RuntimeError("Could not find label column")

    result = result.dropna()
    if max_rows:
        result = result.head(max_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} rows to {output}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public phishing URL dataset")
    parser.add_argument("--output", type=Path, default=Path("data/urls.csv"))
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    try:
        fetch_uci_dataset(args.output, args.max_rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
