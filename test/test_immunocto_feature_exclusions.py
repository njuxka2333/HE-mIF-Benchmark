from data.feature_bags import _load_immunocto_excluded_sample_keys


def test_immunocto_exclusion_csv_maps_relative_and_absolute_paths(tmp_path):
    excluded = tmp_path / "excluded.csv"
    excluded.write_text(
        "image_path\n"
        "CRC29/HE/other_10_20.png\n"
        "/dataset/CRC33/HE/other_30_40.png\n",
        encoding="utf-8",
    )

    assert _load_immunocto_excluded_sample_keys(excluded) == {
        "CRC29/HE/other_10_20.png",
        "CRC33/HE/other_30_40.png",
    }
    assert _load_immunocto_excluded_sample_keys(excluded, split="valid") == {
        "CRC29/HE/other_10_20.png"
    }
