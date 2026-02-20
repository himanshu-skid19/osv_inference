"""
Stage 4: Inference + Evaluation

Pipeline
--------
1. Load fine-tuned encoder from checkpoint.
2. Load enrollment registry (.npz) — cached (mu, Sigma_tilde) per writer.
3. Pre-compute precision matrices (Sigma_tilde^{-1}) for all writers.
4. Batch-encode ALL query images from the test set in one pass.
5. For each writer:
     - genuine queries = genuine_idx[R_ENROLL:]   (held-out from enrollment)
     - forged  queries = all forged indices
     Compute Mahalanobis distance, then EER.
6. Report mean EER +/- std across writers. Save detailed results to JSON.

Usage
-----
    python run_inference.py
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

# ── Shared encoder architecture ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'osv_finetuning'))
from vit import ViTEncoder

from infer    import SignatureVerifier
from evaluate import run_evaluation, aggregate


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = (
    r"C:\Users\Himanshu Singhal\Desktop\BTP"
    r"\osv_finetuning\finetune_runs\run_20260219_102652"
    r"\finetuned_full_weights.pth"
)

# Enrollment registry produced by writer_enrollment/run_enroll.py
ENROLLMENT_PATH = (
    r"C:\Users\Himanshu Singhal\Desktop\BTP"
    r"\writer_enrollment\enrollment_R4_eps0.05_20260220_123659.npz"
)

TEST_DATA_PATH = (
    r"C:\Users\Himanshu Singhal\Desktop\BTP"
    r"\vit_pretraining\deepsigndb_asymmetric_gasf_test.npz"
)

OUTPUT_DIR = r"C:\Users\Himanshu Singhal\Desktop\BTP\inference\results"

# Must match the value used in run_enroll.py
R_ENROLL = 5

# Batch size for encoding queries (reduce if OOM)
ENCODE_BATCH_SIZE = 128

# ══════════════════════════════════════════════════════════════════════════════


def _is_genuine(fname: str) -> bool:
    parts = str(fname).lower().split('_')
    if len(parts) >= 2:
        return parts[1] == 'g'
    return True


def load_encoder(checkpoint_path: str, device: torch.device) -> ViTEncoder:
    run_dir   = os.path.dirname(checkpoint_path)
    args_path = os.path.join(run_dir, 'args.json')
    with open(args_path) as f:
        args = json.load(f)

    encoder = ViTEncoder(
        img_size    = args['img_size'],
        patch_size  = args['patch_size'],
        in_channels = args['in_channels'],
        embed_dim   = args['embed_dim'],
        num_layers  = 12,
        num_heads   = 12,
        mlp_dim     = args['embed_dim'] * 4,
    )
    weights = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(weights['encoder'])
    encoder.eval().to(device)
    print(f"Encoder loaded  (embed_dim={args['embed_dim']}, patch_size={args['patch_size']})")
    return encoder, args['embed_dim']


def load_enrollment(path: str) -> dict:
    if not path.endswith('.npz'):
        path += '.npz'
    data       = np.load(path, allow_pickle=False)
    writer_ids = data['__writer_ids__'].tolist()
    registry   = {}
    for wid in writer_ids:
        registry[wid] = {
            'F_refs'      : data[f'{wid}__F_refs'],
            'mu'          : data[f'{wid}__mu'],
            'sigma_tilde' : data[f'{wid}__sigma_tilde'],
            'sigma_type'  : str(data[f'{wid}__sigma_type']),
            'R'           : int(data[f'{wid}__R']),
        }
    print(f"Enrollment loaded  ({len(registry)} writers)")
    return registry


def load_test_data(data_path: str):
    """
    Load and normalise the test .npz.
    Returns:
        images     : (N, C, H, W) float32, normalised to [-1, 1]
        w2genuine  : writer_id -> list of genuine array indices (full list)
        w2forged   : writer_id -> list of forged  array indices
    """
    data       = np.load(data_path, allow_pickle=True)
    images     = data['gasf_data'].astype(np.float32)
    file_names = data['file_names']

    img_min, img_max = images.min(), images.max()
    if img_max - img_min > 1e-6:
        images = 2.0 * (images - img_min) / (img_max - img_min) - 1.0

    w2genuine: dict = defaultdict(list)
    w2forged:  dict = defaultdict(list)
    for i, fname in enumerate(file_names):
        wid = str(fname).split('_')[0]
        if _is_genuine(fname):
            w2genuine[wid].append(i)
        else:
            w2forged[wid].append(i)

    return images, dict(w2genuine), dict(w2forged)


def encode_all_queries(verifier, images, w2genuine, w2forged, R_enroll):
    """
    Encode every query image (genuine held-out + forged) in one batched pass.

    Returns:
        all_F : (N_total, d) pre-encoded features for all images in the npz
    """
    # Collect all query indices
    query_idx = set()
    for wid, gen_idx in w2genuine.items():
        if wid not in verifier.registry:
            continue
        query_idx.update(gen_idx[R_enroll:])
        query_idx.update(w2forged.get(wid, []))

    query_idx = sorted(query_idx)
    print(f"Encoding {len(query_idx)} query images ...")

    imgs_tensor = torch.from_numpy(images[query_idx])            # (Q, C, H, W)
    F_queries   = verifier.encode_batch(imgs_tensor)             # (Q, d)

    # Map back from query position to original array index
    all_F = np.zeros((len(images), verifier.embed_dim), dtype=np.float32)
    for pos, orig_idx in enumerate(query_idx):
        all_F[orig_idx] = F_queries[pos]

    return all_F


def print_summary(summary: dict, per_writer: list):
    print()
    print("=" * 55)
    print("  EVALUATION RESULTS  (DeepSignDB test set)")
    print("=" * 55)
    print(f"  Writers evaluated : {summary['n_writers']}")
    print(f"  Mean EER          : {summary['mean_eer']*100:.2f} %")
    print(f"  Std  EER          : {summary['std_eer']*100:.2f} %")
    print(f"  Median EER        : {summary['median_eer']*100:.2f} %")
    print(f"  Min  EER          : {summary['min_eer']*100:.2f} %")
    print(f"  Max  EER          : {summary['max_eer']*100:.2f} %")
    print("=" * 55)

    # Top-5 best and worst writers
    sorted_w = sorted(per_writer, key=lambda r: r['eer'])
    print("\n  Best 5 writers:")
    for r in sorted_w[:5]:
        print(f"    {r['writer_id']}  EER={r['eer']*100:.2f}%  "
              f"(d_gen={r['mean_genuine_dist']:.1f}, d_forg={r['mean_forged_dist']:.1f})")
    print("\n  Worst 5 writers:")
    for r in sorted_w[-5:]:
        print(f"    {r['writer_id']}  EER={r['eer']*100:.2f}%  "
              f"(d_gen={r['mean_genuine_dist']:.1f}, d_forg={r['mean_forged_dist']:.1f})")


def plot_results(per_writer: list, output_dir: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        eers = [r['eer'] * 100 for r in per_writer]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram of per-writer EERs
        axes[0].hist(eers, bins=30, edgecolor='black', color='steelblue')
        axes[0].axvline(np.mean(eers), color='red', linestyle='--',
                        label=f'Mean = {np.mean(eers):.2f}%')
        axes[0].set_xlabel('EER (%)')
        axes[0].set_ylabel('Number of writers')
        axes[0].set_title('Per-writer EER distribution')
        axes[0].legend()

        # Scatter: mean genuine dist vs mean forged dist
        d_gen  = [r['mean_genuine_dist'] for r in per_writer]
        d_forg = [r['mean_forged_dist']  for r in per_writer]
        axes[1].scatter(d_gen, d_forg, alpha=0.5, s=15, color='steelblue')
        lim = max(max(d_gen), max(d_forg)) * 1.05
        axes[1].plot([0, lim], [0, lim], 'r--', label='d_gen = d_forg')
        axes[1].set_xlabel('Mean genuine Mahalanobis distance')
        axes[1].set_ylabel('Mean forged Mahalanobis distance')
        axes[1].set_title('Genuine vs forged distances per writer')
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'eer_results.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n  Plot saved to: {plot_path}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load components ───────────────────────────────────────────────────────
    print("\n[1] Loading encoder ...")
    encoder, embed_dim = load_encoder(CHECKPOINT_PATH, device)

    print("\n[2] Loading enrollment registry ...")
    registry = load_enrollment(ENROLLMENT_PATH)

    print("\n[3] Building verifier (precomputing precision matrices) ...")
    verifier = SignatureVerifier(
        encoder    = encoder,
        registry   = registry,
        embed_dim  = embed_dim,
        device     = device,
        batch_size = ENCODE_BATCH_SIZE,
    )

    # ── Load test data ────────────────────────────────────────────────────────
    print("\n[4] Loading test data ...")
    images, w2genuine, w2forged = load_test_data(TEST_DATA_PATH)
    n_query_gen  = sum(max(0, len(v) - R_ENROLL) for v in w2genuine.values())
    n_query_forg = sum(len(v) for v in w2forged.values())
    print(f"    Genuine query samples : {n_query_gen}")
    print(f"    Forged  query samples : {n_query_forg}")

    # ── Encode all queries in one pass ────────────────────────────────────────
    print("\n[5] Encoding all query images ...")
    all_F = encode_all_queries(verifier, images, w2genuine, w2forged, R_ENROLL)

    # ── Per-writer evaluation ─────────────────────────────────────────────────
    print("\n[6] Computing Mahalanobis distances and EER per writer ...")
    per_writer = run_evaluation(
        verifier   = verifier,
        images     = images,
        w2genuine  = w2genuine,
        w2forged   = w2forged,
        R_enroll   = R_ENROLL,
        all_F      = all_F,
    )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = aggregate(per_writer)
    print_summary(summary, per_writer)

    # ── Save ──────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = {
        'timestamp'   : timestamp,
        'checkpoint'  : CHECKPOINT_PATH,
        'enrollment'  : ENROLLMENT_PATH,
        'R_enroll'    : R_ENROLL,
        'dataset'     : 'DeepSignDB',
        'summary'     : summary,
        'per_writer'  : sorted(per_writer, key=lambda r: r['writer_id']),
    }
    json_path = os.path.join(OUTPUT_DIR, f'results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to: {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(per_writer, OUTPUT_DIR)


if __name__ == '__main__':
    main()
