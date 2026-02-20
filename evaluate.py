"""
Evaluation utilities for OSV inference.

  compute_eer(genuine_dists, forged_dists)
      -> (eer, tau_star)

  run_evaluation(verifier, images, w2genuine, w2forged, R_enroll)
      -> list of per-writer result dicts

  aggregate(results)
      -> summary dict  { mean_eer, std_eer, median_eer, min_eer, max_eer }
"""

from typing import Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# EER
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(
    genuine_dists: np.ndarray,
    forged_dists:  np.ndarray,
) -> Tuple[float, float]:
    """
    Find the Equal Error Rate operating point.

      FRR(tau) = P(d_M > tau  | genuine)   — genuine rejected
      FAR(tau) = P(d_M <= tau | forged)    — forged accepted
      tau* = argmin |FRR(tau) - FAR(tau)|
      EER  = ( FRR(tau*) + FAR(tau*) ) / 2

    Args:
        genuine_dists : (N_g,) Mahalanobis distances for genuine queries.
        forged_dists  : (N_f,) Mahalanobis distances for forged queries.

    Returns:
        eer      : float in [0, 1].
        tau_star : float threshold.
    """
    if len(genuine_dists) == 0 or len(forged_dists) == 0:
        return float('nan'), float('nan')

    # All unique observed distances as candidate thresholds
    thresholds = np.sort(np.unique(
        np.concatenate([genuine_dists, forged_dists])
    ))

    # Vectorised computation over all thresholds at once
    # genuine_dists: (N_g,)  thresholds: (T,)
    frr = (genuine_dists[:, None] > thresholds[None, :]).mean(axis=0)   # (T,)
    far = (forged_dists[:, None]  <= thresholds[None, :]).mean(axis=0)  # (T,)

    idx      = int(np.argmin(np.abs(frr - far)))
    eer      = float((frr[idx] + far[idx]) / 2.0)
    tau_star = float(thresholds[idx])

    return eer, tau_star


# ─────────────────────────────────────────────────────────────────────────────
# Per-writer evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    verifier,           # SignatureVerifier
    images:   np.ndarray,
    w2genuine: Dict[str, List[int]],
    w2forged:  Dict[str, List[int]],
    R_enroll:  int,
    all_F:     np.ndarray = None,   # optional pre-encoded features (N_total, d)
) -> List[dict]:
    """
    Evaluate every enrolled writer.

    For each writer:
      - Genuine queries = genuine indices[R_enroll:]  (held-out references)
      - Forged  queries = all forged indices

    Computes per-writer Mahalanobis distances, then EER.

    Args:
        verifier   : SignatureVerifier (with enrollment registry loaded).
        images     : (N_total, C, H, W) float32 numpy array, normalised.
        w2genuine  : writer_id -> list of genuine array indices (full list).
        w2forged   : writer_id -> list of forged  array indices.
        R_enroll   : number of genuine samples used for enrollment (excluded).
        all_F      : (N_total, d) pre-encoded GAP features, or None to encode
                     on the fly per writer.

    Returns:
        List of dicts, one per writer:
            writer_id, n_genuine_queries, n_forged_queries,
            eer, tau_star,
            mean_genuine_dist, mean_forged_dist
    """
    import torch

    results = []
    enrolled_writers = set(verifier.registry.keys())

    for writer_id, gen_idx in w2genuine.items():

        if writer_id not in enrolled_writers:
            continue

        # ── Held-out genuine queries (skip enrollment samples) ────────────────
        query_gen_idx  = gen_idx[R_enroll:]
        query_forg_idx = w2forged.get(writer_id, [])

        if len(query_gen_idx) == 0 or len(query_forg_idx) == 0:
            continue

        # ── Get GAP features ─────────────────────────────────────────────────
        if all_F is not None:
            F_gen  = all_F[query_gen_idx]
            F_forg = all_F[query_forg_idx]
        else:
            imgs_gen  = torch.from_numpy(images[query_gen_idx])
            imgs_forg = torch.from_numpy(images[query_forg_idx])
            F_gen     = verifier.encode_batch(imgs_gen)
            F_forg    = verifier.encode_batch(imgs_forg)

        # ── Mahalanobis distances ─────────────────────────────────────────────
        d_gen  = verifier.mahalanobis(F_gen,  writer_id)   # (N_g,)
        d_forg = verifier.mahalanobis(F_forg, writer_id)   # (N_f,)

        # ── EER ──────────────────────────────────────────────────────────────
        eer, tau_star = compute_eer(d_gen, d_forg)

        results.append({
            'writer_id'         : writer_id,
            'n_genuine_queries' : len(d_gen),
            'n_forged_queries'  : len(d_forg),
            'eer'               : eer,
            'tau_star'          : tau_star,
            'mean_genuine_dist' : float(d_gen.mean()),
            'mean_forged_dist'  : float(d_forg.mean()),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: List[dict]) -> dict:
    """
    Compute mean EER +/- std and other summary statistics across writers.
    NaN writers (no genuine or forged queries) are excluded.
    """
    eers = np.array([r['eer'] for r in results])
    eers = eers[~np.isnan(eers)]

    return {
        'n_writers'  : len(eers),
        'mean_eer'   : float(eers.mean()),
        'std_eer'    : float(eers.std()),
        'median_eer' : float(np.median(eers)),
        'min_eer'    : float(eers.min()),
        'max_eer'    : float(eers.max()),
    }
