"""Build local patch files from raw images. Nothing pixel-sized lives in git.

    python -m preprocessing.build --dataset tcga_coad
    python -m preprocessing.build --dataset tcga_luad
    python -m preprocessing.build --dataset hooknet_tls
    python -m preprocessing.build --dataset cell_niches
    python -m preprocessing.build --dataset lizard
    python -m preprocessing.build --dataset immunocto

Prognosis writes CLAM ``patches/{slide}.h5`` coords.
TLS writes PNG+mask from the frozen ``splits/tls_region_segmentation/hooknet_tls/patches.csv`` crop boxes.
Cell-niches writes paired HE/mIF tiles from the frozen VS catalog.
Lizard converts official PNG/.mat labels into PathoCell-style HDF.
Immunocto writes the webdataset shards used by feature extraction.
PathoCell and NuCLS use the released patch files; there is nothing to build.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

from data.hooknet_tls.planning import load_selected_tiles_csv
from data.hooknet_tls.physical_grid import materialize_physical_record
from data.hooknet_tls.annotations import load_official_wsa
from data.loader import get_dataset_config, load_split_samples
from preprocessing.wsi.coords import coords_for_slide, load_wsi_patch_protocol, write_coords_h5
from utils.paths import SPLITS_ROOT, cell_niches_raw_root, datasets_root

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--slide-id", default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--data-config", default=None)
    return p.parse_args(argv)


def _build_wsi_coords(dataset: str, args: argparse.Namespace) -> None:
    cfg = get_dataset_config(dataset, args.data_config)
    proto = load_wsi_patch_protocol()
    samples = load_split_samples(cfg)
    seen: set[str] = set()
    out_dir = cfg.coords_dir()
    skip = bool(args.skip_existing) and not bool(args.overwrite)
    n_write = n_skip = n_miss = 0
    for sample in samples:
        if args.slide_id and sample.sample_id != args.slide_id:
            continue
        if sample.sample_id in seen:
            continue
        seen.add(sample.sample_id)
        dest = out_dir / f"{sample.sample_id}.h5"
        if skip and dest.is_file():
            n_skip += 1
            continue
        wsi = sample.image_path
        if wsi is None or not Path(wsi).is_file():
            print(f"missing WSI for {sample.sample_id}", flush=True)
            n_miss += 1
            continue
        coords, attr = coords_for_slide(Path(wsi), proto)
        attr["name"] = sample.sample_id
        write_coords_h5(dest, coords, attr)
        n_write += 1
        print(f"{sample.sample_id}: {len(coords)} coords -> {dest}", flush=True)
    print(f"wrote={n_write} skipped={n_skip} missing_wsi={n_miss} out={out_dir}")


def _build_tls(args: argparse.Namespace) -> None:
    from wholeslidedata.image.wholeslideimage import WholeSlideImage

    cfg = get_dataset_config("hooknet_tls", args.data_config)
    catalog = SPLITS_ROOT / "tls_region_segmentation" / "hooknet_tls" / "patches.csv"
    if not catalog.is_file():
        raise FileNotFoundError(f"frozen TLS catalog missing: {catalog}")
    records = load_selected_tiles_csv(catalog)
    if args.slide_id:
        records = [r for r in records if r.slide_id == args.slide_id]
    samples = {s.sample_id: s for s in load_split_samples(cfg)}
    out = datasets_root() / "HookNet_TLS_fixed_grid_ps1024"
    skip = bool(args.skip_existing) and not bool(args.overwrite)
    by_slide: dict[str, list] = defaultdict(list)
    for rec in records:
        by_slide[rec.slide_id].append(rec)
    xml_dir = cfg.annotations_dir()
    n_write = n_skip = 0
    for slide_id, rows in sorted(by_slide.items()):
        sample = samples.get(slide_id)
        wsi_path = sample.image_path if sample else None
        if wsi_path is None or not Path(wsi_path).is_file():
            print(f"missing WSI for {slide_id}", flush=True)
            continue
        xml_path = None if xml_dir is None else xml_dir / f"{slide_id}.xml"
        wsi = WholeSlideImage(str(wsi_path), backend="openslide")
        wsa = load_official_wsa(xml_path, backend="openslide") if xml_path else None
        if wsa is None:
            wsi.close()
            raise RuntimeError(f"Missing WSA for {slide_id}")
        try:
            source_mpp = float(wsi.spacings[0])
            for rec in rows:
                stem = f"{rec.slide_id}_ix{rec.grid_ix}_iy{rec.grid_iy}.png"
                image_path = out / rec.split / "images" / stem
                if skip and image_path.is_file():
                    n_skip += 1
                    continue
                materialize_physical_record(
                    rec,
                    wsi=wsi,
                    wsa=wsa,
                    source_mpp=rec.source_mpp if rec.source_mpp > 0 else source_mpp,
                    reference_mpp=0.25,
                    images_dir=out / rec.split / "images",
                    masks_dir=out / rec.split / "masks",
                )
                n_write += 1
        finally:
            wsi.close()
        print(f"{slide_id}: materialized {len(rows)} tiles", flush=True)
    dest_catalog = out / "patches.csv"
    if not dest_catalog.is_file() or args.overwrite:
        dest_catalog.write_bytes(catalog.read_bytes())
    print(f"wrote={n_write} skipped={n_skip} catalog={dest_catalog}")


def _build_cell_niches(args: argparse.Namespace) -> None:
    from preprocessing.cell_niches.extract import FilterConfig, extract_cohort

    out = datasets_root() / "cell_niches_paired_patches"
    frozen = SPLITS_ROOT / "virtual_staining" / "cell_niches" / "patches.csv"
    root = cell_niches_raw_root()
    extract_cohort(root, out, FilterConfig(), write_patches=True, fov_ids=None)
    if frozen.is_file() and not (out / "patches.csv").is_file():
        (out / "patches.csv").write_bytes(frozen.read_bytes())
    print(f"cell_niches out={out}")


def _run_script(filename: str, argv: list[str]) -> None:
    path = _SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = sys.argv
    sys.argv = [filename, *argv]
    try:
        mod.main()
    finally:
        sys.argv = old


def _build_lizard(args: argparse.Namespace) -> None:
    cfg = get_dataset_config("lizard", args.data_config)
    forwarded = ["--dataset-root", str(cfg.dataset_dir)]
    if args.skip_existing and not args.overwrite:
        forwarded.append("--skip-existing")
    _run_script("build_lizard_pathocell_hdf.py", forwarded)


def _build_immunocto(args: argparse.Namespace) -> None:
    cfg = get_dataset_config("immunocto", args.data_config)
    forwarded = [
        "--root",
        str(cfg.dataset_dir),
        "--output-root",
        str(datasets_root() / "immunocto_wds"),
    ]
    _run_script("build_immunocto_webdataset.py", forwarded)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    if args.dataset in {"tcga_coad", "tcga_luad"}:
        _build_wsi_coords(args.dataset, args)
        return
    if args.dataset in {"hooknet_tls", "tls"}:
        _build_tls(args)
        return
    if args.dataset in {"cell_niches", "virtual_staining"}:
        _build_cell_niches(args)
        return
    if args.dataset == "lizard":
        _build_lizard(args)
        return
    if args.dataset == "immunocto":
        _build_immunocto(args)
        return
    if args.dataset in {"pathocell", "nucls"}:
        print(f"{args.dataset}: released patch files; nothing to build")
        return
    raise SystemExit(
        f"unknown dataset {args.dataset!r}; "
        "use tcga_coad, tcga_luad, hooknet_tls, cell_niches, lizard, immunocto, "
        "pathocell, or nucls"
    )


if __name__ == "__main__":
    main()
