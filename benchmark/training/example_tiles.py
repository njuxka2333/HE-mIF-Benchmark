"""Task-level example-tile figures, called by ``training.pipeline`` after postprocess."""

from __future__ import annotations


def render_task_example_tiles(task: str, args) -> None:
    models = tuple(args.models) if getattr(args, "models", None) else None
    device = getattr(args, "device", None)
    if task == "virtual_staining":
        from mif.example_tile import render_vstain_example_tile

        print("\n--- example tile: virtual staining CD3 ---", flush=True)
        dest = render_vstain_example_tile(models=models, device=device)
        print(f"    wrote {dest}", flush=True)
        return
    if task == "cell_semantic_segmentation":
        from training.pathocell_example_tile import render_pathocell_example_tile

        print("\n--- example tile: PathoCell encoder crop ---", flush=True)
        dest = render_pathocell_example_tile(models=models, device=device, force=True)
        print(f"    wrote {dest}", flush=True)
        return
    if task == "tls_region_segmentation":
        from training.tls_example_tile import render_tls_example_tile

        print("\n--- example tile: TLS encoder crop ---", flush=True)
        dest = render_tls_example_tile(models=models, device=device, force=True)
        print(f"    wrote {dest}", flush=True)
        return
