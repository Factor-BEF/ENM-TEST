#!/usr/bin/env python3
"""Download the fixed GBIF historical occurrence archive for Vulpes vulpes.

Dataset DOI: 10.15468/dl.ktotdn
GBIF download key: 0008885-200221144449610
Temporal filter in archived dataset: 1980-2017
The archived query used hasCoordinate=true and hasGeospatialIssue=false.
"""

from pathlib import Path
import requests

DOWNLOAD_KEY = "0008885-200221144449610"
URLS = [
    f"https://api.gbif.org/v1/occurrence/download/request/{DOWNLOAD_KEY}.zip",
    f"https://api.gbif.org/occurrence/download/request/{DOWNLOAD_KEY}.zip",
]
OUT = Path(__file__).with_name("Vulpes_vulpes_GBIF_1980_2017.zip")


def download() -> None:
    last_error = None
    for url in URLS:
        try:
            print(f"Downloading: {url}")
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with OUT.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            print(f"Saved: {OUT.resolve()}")
            print(f"Size: {OUT.stat().st_size / (1024 ** 2):.1f} MB")
            return
        except requests.RequestException as exc:
            last_error = exc
            if OUT.exists():
                OUT.unlink()

    raise RuntimeError(f"GBIF archive download failed: {last_error}")


if __name__ == "__main__":
    download()
