from __future__ import annotations

import csv
import gzip
from pathlib import Path

from data.loader import open_csv_table, resolve_table_path


def test_resolve_table_path_falls_back_to_gz(tmp_path: Path):
    csv_path = tmp_path / "table.csv"
    gz_path = tmp_path / "table.csv.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write("image_path\na.png\n")
    assert resolve_table_path(csv_path) == gz_path
    with open_csv_table(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"image_path": "a.png"}]


def test_resolve_table_path_prefers_plain_csv(tmp_path: Path):
    csv_path = tmp_path / "table.csv"
    gz_path = tmp_path / "table.csv.gz"
    csv_path.write_text("image_path\nb.png\n", encoding="utf-8")
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write("image_path\na.png\n")
    assert resolve_table_path(csv_path) == csv_path
    with open_csv_table(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"image_path": "b.png"}]
