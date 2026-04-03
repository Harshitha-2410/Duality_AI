"""
=============================================================================
  Duality AI — Offroad Desert Segmentation  |  TEST / INFERENCE SCRIPT
  Loads best_model.pth, runs on testImages/, produces:
    • Colorized segmentation masks
    • Side-by-side comparison PNGs
    • IoU score (if ground truth available)
    • test_results.json summary
=============================================================================
Usage:
    conda activate EDU
    python test.py --checkpoint runs/<timestamp>/checkpoints/best_model.pth
    python test.py --checkpoint best_model.pth --gt-dir data/val/segmentation
=============================================================================
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ─────────────────────────── CLASS CONFIG ─────────────────────────────────
CLASS_MAP = {
    100  : (0,  "Trees"),
    200  : (1,  "Lush Bushes"),
    300  : (2,  "Dry Grass"),
    500  : (3,  "Dry Bushes"),
    550  : (4,  "Ground Clutter"),
    600  : (5,  "Flowers"),
    700  : (6,  "Logs"),
    800  : (7,  "Rocks"),
    7100 : (8,  "Landscape"),
    10000: (9,  "Sky"),
}
NUM_CLASSES = 10
CLASS_NAMES = [v[1] for v in sorted(CLASS_MAP.values())]

# High-contrast, visually distinct colors for each class
CLASS_COLORS = np.array([
    [ 34, 139,  34],   # Trees        — forest green
    [  0, 200, 120],   # Lush Bushes  — emerald
    [210, 180, 140],   # Dry Grass    — tan
    [160, 100,  40],   # Dry Bushes   — brown
    [120,  70,  30],   # Gnd Clutter  — dark brown
    [255, 215,   0],   # Flowers      — gold
    [101,  67,  33],   # Logs         — dark wood
    [130, 130, 130],   # Rocks        — slate grey
    [194, 164, 117],   # Landscape    — sandy
    [135, 206, 235],   # Sky          — sky blue
], dtype=np.uint8)

BG = "#0d1117"         # dark background for figures


# ─────────────────────────── MODEL LOADER ─────────────────────────────────
def load_model(checkpoint: str, device: torch.device):
    model = models.segmentation.deeplabv3_resnet101(weights=None)
    model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"  ✦ Loaded checkpoint: {checkpoint}")
    if "epoch" in ckpt:
        print(f"  ✦ Trained for {ckpt['epoch']} epochs  |  Val IoU = {ckpt.get('iou', '?'):.4f}")
    return model


# ─────────────────────────── PREPROCESSING ────────────────────────────────
_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess(img_path: Path, size: int):
    img      = Image.open(img_path).convert("RGB")
    orig_wh  = img.size
    resized  = img.resize((size, size), Image.BILINEAR)
    tensor   = _tf(resized).unsqueeze(0)
    return tensor, orig_wh, img


# ─────────────────────────── COLORIZE ─────────────────────────────────────
def colorize(mask: np.ndarray) -> np.ndarray:
    """Map class indices → RGB color array."""
    h, w = mask.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(NUM_CLASSES):
        out[mask == i] = CLASS_COLORS[i]
    return out


# ─────────────────────────── IoU ──────────────────────────────────────────
def build_lut():
    lut = np.zeros(65536, dtype=np.uint8)
    for raw, (idx, _) in CLASS_MAP.items():
        if raw < 65536:
            lut[raw] = idx
    return lut

_LUT = build_lut()

def load_gt_mask(path: Path, target_wh) -> np.ndarray:
    raw = np.array(Image.open(path))
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    raw  = raw.astype(np.uint16)
    mask = _LUT[raw]
    return np.array(Image.fromarray(mask).resize(target_wh, Image.NEAREST))

def compute_iou(pred: np.ndarray, gt: np.ndarray):
    ious = {}
    p, g = pred.flatten(), gt.flatten()
    for c in range(NUM_CLASSES):
        pi, gi  = p == c, g == c
        inter   = (pi & gi).sum()
        union   = (pi | gi).sum()
        ious[CLASS_NAMES[c]] = float(inter) / float(union) if union > 0 else None
    valid       = [v for v in ious.values() if v is not None]
    ious["mean_iou"] = float(np.mean(valid)) if valid else 0.0
    return ious


# ─────────────────────────── VISUALISATION ────────────────────────────────
def save_comparison(orig_img: Image.Image, color_mask: np.ndarray,
                    save_path: Path, fname: str, iou: float):
    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    ax0.imshow(orig_img);        ax0.set_title("Original RGB",  color="white", fontsize=12, pad=8)
    ax1.imshow(color_mask);      ax1.set_title(f"Predicted Segmentation  |  IoU: {iou:.3f}",
                                               color="white", fontsize=12, pad=8)
    for ax in (ax0, ax1):
        ax.axis("off")
        ax.set_facecolor(BG)

    patches = [mpatches.Patch(facecolor=CLASS_COLORS[i] / 255, label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               facecolor="#161b22", labelcolor="white", framealpha=0.9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Duality AI Desert Segmentation — {fname}",
                 color="white", fontsize=13, y=1.01)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def save_iou_chart(iou_dict: dict, save_path: Path):
    names  = CLASS_NAMES
    values = [iou_dict.get(n, 0) or 0 for n in names]
    colors = [CLASS_COLORS[i] / 255 for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(names, values, color=colors, edgecolor="#ffffff22", linewidth=0.5)
    ax.axvline(iou_dict.get("mean_iou", 0), color="#00f5ff", linewidth=1.5,
               linestyle="--", label=f"Mean IoU = {iou_dict.get('mean_iou', 0):.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("IoU Score", color="white"); ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    for spine in ax.spines.values(): spine.set_color("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=9)
    plt.title("Per-Class IoU Scores", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=BG)
    plt.close(fig)


# ─────────────────────────── MAIN TEST LOOP ───────────────────────────────
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  Duality AI Desert Segmentation — Test / Inference")
    print(f"  Device     : {device}")
    print(f"  Test dir   : {args.test_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'='*65}\n")

    model = load_model(args.checkpoint, device)

    test_dir = Path(args.test_dir)
    out_dir  = Path(args.output_dir)
    masks_dir = out_dir / "masks"
    color_dir = out_dir / "colorized"
    comp_dir  = out_dir / "comparisons"
    for d in (masks_dir, color_dir, comp_dir):
        d.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        p for p in test_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    print(f"  Found {len(img_paths)} test images\n")

    all_times, results = [], []
    agg_iou_dict = {n: [] for n in CLASS_NAMES}

    for img_path in img_paths:
        t0 = time.time()
        tensor, orig_wh, orig_img = preprocess(img_path, args.size)
        tensor = tensor.to(device)

        with torch.no_grad():
            out = model(tensor)
        pred_sm = out["out"].softmax(dim=1)
        pred    = pred_sm.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize prediction back to original size
        pred_full = np.array(
            Image.fromarray(pred).resize(orig_wh, Image.NEAREST)
        )
        elapsed_ms = (time.time() - t0) * 1000
        all_times.append(elapsed_ms)

        # Save raw mask
        Image.fromarray(pred_full).save(masks_dir / f"{img_path.stem}_mask.png")

        # Colorize
        color_mask = colorize(pred_full)
        Image.fromarray(color_mask).save(color_dir / f"{img_path.stem}_color.png")

        # Compute IoU if GT available
        iou_result = {}
        if args.gt_dir:
            gt_path = Path(args.gt_dir) / (img_path.stem + ".png")
            if not gt_path.exists():
                gt_path = Path(args.gt_dir) / img_path.name
            if gt_path.exists():
                gt = load_gt_mask(gt_path, orig_wh)
                iou_result = compute_iou(pred_full, gt)
                for n in CLASS_NAMES:
                    if iou_result.get(n) is not None:
                        agg_iou_dict[n].append(iou_result[n])

        miou = iou_result.get("mean_iou", -1)
        save_comparison(orig_img, color_mask,
                        comp_dir / f"{img_path.stem}_compare.png",
                        img_path.name, max(miou, 0))

        results.append({
            "image": img_path.name,
            "inference_ms": round(elapsed_ms, 2),
            "iou": iou_result,
        })
        status = f"IoU={miou:.4f}" if miou >= 0 else "no GT"
        print(f"  ✓ {img_path.name:<40} {elapsed_ms:6.1f}ms  {status}")

    # ── Summary ───────────────────────────────────────────────────────────
    avg_ms   = float(np.mean(all_times))
    per_mean = {n: float(np.mean(v)) if v else None for n, v in agg_iou_dict.items()}
    valid    = [v for v in per_mean.values() if v is not None]
    overall  = float(np.mean(valid)) if valid else None

    summary = {
        "total_images"     : len(img_paths),
        "avg_inference_ms" : round(avg_ms, 2),
        "inference_ok"     : avg_ms < 50,
        "overall_mean_iou" : round(overall, 4) if overall else None,
        "per_class_iou"    : {k: (round(v, 4) if v else None)
                              for k, v in per_mean.items()},
        "results"          : results,
    }

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    if overall is not None:
        all_iou_entry = dict(per_mean)
        all_iou_entry["mean_iou"] = overall
        save_iou_chart(all_iou_entry, out_dir / "iou_chart.png")

    # ── Print report ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TEST COMPLETE")
    print(f"  Images processed : {len(img_paths)}")
    print(f"  Avg inference    : {avg_ms:.1f} ms  "
          f"({'✓ <50ms' if avg_ms < 50 else '✗ exceeds 50ms target'})")
    if overall:
        print(f"  Overall IoU      : {overall:.4f}")
        print(f"\n  Per-class IoU:")
        for name, val in per_mean.items():
            bar = "█" * int((val or 0) * 20)
            print(f"    {name:<20} {val:.3f if val else 'N/A '}  {bar}")
    print(f"\n  Results → {out_dir}")
    print(f"{'='*65}\n")


# ─────────────────────────── ENTRY POINT ──────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Duality AI Desert Segmentation – Test")
    p.add_argument("--checkpoint",  required=True,          help="Path to best_model.pth")
    p.add_argument("--test-dir",    default="Offroad_Segmentation_testImages/Color_Images", help="Folder with RGB test images")
    p.add_argument("--output-dir",  default="./outputs",    help="Where to save results")
    p.add_argument("--gt-dir",      default="Offroad_Segmentation_testImages/Segmentation",           help="Ground truth masks (optional, for IoU)")
    p.add_argument("--size",        type=int, default=512,  help="Inference resize resolution")
    args = p.parse_args()
    test(args)