"""Downloads the three raw datasets into data/raw/. Run: python -m inclination.load.download"""
from pathlib import Path
from urllib.parse import quote

from inclination.config import load_config, path  # first: sets HF_HOME before the HF import

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfFileSystem, hf_hub_download


def download_craigslist(src: dict, out_dir: Path) -> None:
    for filename in src["files"]:
        hf_hub_download(
            src["hf_repo"],
            filename,
            repo_type="dataset",
            revision=src["hf_revision"],
            local_dir=out_dir,
        )
        print(f"  craigslist: {filename}")


def download_p4g(src: dict, out_dir: Path, timeout: int) -> None:
    for filename in src["files"]:
        target = out_dir / filename
        if target.exists():
            print(f"  p4g: {filename} (already present)")
            continue
        response = requests.get(f"{src['base_url']}/{filename}", timeout=timeout)
        response.raise_for_status()
        target.write_bytes(response.content)
        print(f"  p4g: {filename} ({len(response.content):,} bytes)")


def download_saas(src: dict, out_dir: Path) -> None:
    """Reads only the non-embedding columns of each remote shard (parquet column projection)."""
    fs = HfFileSystem()
    repo = f"datasets/{src['hf_repo']}@{quote(src['hf_revision'], safe='')}"
    for shard in sorted(fs.glob(f"{repo}/{src['shard_glob']}")):
        target = out_dir / Path(shard).name
        if target.exists():
            print(f"  saas: {target.name} (already present)")
            continue
        with fs.open(shard, "rb") as f:
            parquet_file = pq.ParquetFile(f)
            columns = [
                name
                for name in parquet_file.schema_arrow.names
                if not name.startswith(src["drop_column_prefix"])
            ]
            table = parquet_file.read(columns=columns)
        pq.write_table(table, target)
        print(f"  saas: {target.name} ({table.num_rows:,} rows, {len(columns)} columns)")


def main() -> None:
    cfg = load_config()["data"]
    sources = cfg["sources"]
    raw = path("raw")
    for name in cfg["datasets"]:
        (raw / name).mkdir(parents=True, exist_ok=True)
    print("Downloading raw datasets...")
    download_craigslist(sources["craigslist"], raw / "craigslist")
    download_p4g(sources["p4g"], raw / "p4g", cfg["download_timeout_s"])
    download_saas(sources["saas"], raw / "saas")
    print(f"Done. Raw data is in {raw}")


if __name__ == "__main__":
    main()
