"""
Stage 4: Inference — SignatureVerifier

Loads a pre-built enrollment registry and provides:
  - Batch GAP encoding of query images via the fine-tuned encoder.
  - Mahalanobis distance computation against enrolled writer statistics.

Sigma_tilde^{-1} is precomputed once at init for all enrolled writers.
  - 'full'  sigma_tilde : inverted via Cholesky decomposition.
  - 'diag'  sigma_tilde : inverse = 1 / sigma_tilde  (element-wise).
"""

import numpy as np
import torch
import torch.nn as nn


class SignatureVerifier:
    """
    Wraps the fine-tuned encoder and the enrollment registry to perform
    online signature verification via Mahalanobis distance.

    Args:
        encoder    : fine-tuned ViTEncoder in eval mode.
        registry   : dict  writer_id -> {'mu', 'sigma_tilde', 'sigma_type', 'R', 'F_refs'}
        embed_dim  : encoder hidden dimension d.
        device     : torch device for encoding.
        batch_size : images per forward pass.
    """

    def __init__(
        self,
        encoder: nn.Module,
        registry: dict,
        embed_dim: int = 768,
        device: torch.device = torch.device('cpu'),
        batch_size: int = 64,
    ):
        self.encoder    = encoder
        self.registry   = registry
        self.embed_dim  = embed_dim
        self.device     = device
        self.batch_size = batch_size

        # Precompute precision matrices (Sigma_tilde^{-1}) for all writers
        print("Pre-computing precision matrices ...")
        self._precision = {}     # writer_id -> ('full', L_inv) | ('diag', inv_diag)
        self._precompute_inverses()
        print(f"  Done — {len(self._precision)} writers ready.")

    # ── Precision precomputation ──────────────────────────────────────────────

    def _precompute_inverses(self):
        for wid, rec in self.registry.items():
            sigma_type  = rec['sigma_type']
            sigma_tilde = rec['sigma_tilde']

            if sigma_type == 'full':
                # Cholesky:  Sigma = L L^T  =>  Sigma^{-1} = L^{-T} L^{-1}
                # d_M = || L^{-1} delta ||_2
                # Precompute L^{-1} to make per-query cost O(d^2).
                try:
                    L     = np.linalg.cholesky(sigma_tilde.astype(np.float64))
                    L_inv = np.linalg.inv(L)           # (d, d)  lower-triangular inverse
                    self._precision[wid] = ('chol', L_inv.astype(np.float32))
                except np.linalg.LinAlgError:
                    # Fallback to direct inverse if Cholesky fails
                    sig_inv = np.linalg.inv(sigma_tilde.astype(np.float64))
                    self._precision[wid] = ('inv', sig_inv.astype(np.float32))

            else:  # 'diag'
                inv_diag = (1.0 / sigma_tilde).astype(np.float32)   # (d,)
                self._precision[wid] = ('diag', inv_diag)

    # ── Encoding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encode images and return GAP patch features.

        Args:
            images : (N, C, H, W) float32 tensor, normalised to [-1, 1].

        Returns:
            F : (N, d) float32 numpy array.
                F_q = (1/L) sum_{l=1}^{L} Z_q^{(l)}   [patch tokens only]
        """
        self.encoder.eval()
        all_F = []
        for start in range(0, len(images), self.batch_size):
            batch  = images[start : start + self.batch_size].to(self.device)
            tokens = self.encoder(batch, return_all_tokens=True)  # (B, L+1, d)
            Z_patch = tokens[:, 1:, :]                            # (B, L, d)
            F       = Z_patch.mean(dim=1)                         # (B, d)
            all_F.append(F.cpu().float().numpy())
        return np.concatenate(all_F, axis=0)                      # (N, d)

    # ── Mahalanobis distance ─────────────────────────────────────────────────

    def mahalanobis(self, F_q: np.ndarray, writer_id: str) -> np.ndarray:
        """
        Compute Mahalanobis distance from each query feature to writer prototype.

        d_M(x_q, w*) = sqrt( (F_q - mu)^T  Sigma_tilde^{-1}  (F_q - mu) )

        Args:
            F_q       : (N, d) float32 array of query GAP features.
            writer_id : enrolled writer.

        Returns:
            distances : (N,) float32 array.
        """
        rec   = self.registry[writer_id]
        mu    = rec['mu']                          # (d,)
        delta = (F_q - mu[None, :]).astype(np.float32)   # (N, d)

        prec_type, prec_data = self._precision[writer_id]

        if prec_type == 'chol':
            # d_M = || L^{-1} delta^T ||_2  (column-wise L2 norm)
            # prec_data = L^{-1}  (d, d)
            # x = L^{-1} @ delta^T  ->  (d, N)
            x         = prec_data @ delta.T          # (d, N)
            distances = np.sqrt((x ** 2).sum(axis=0))  # (N,)

        elif prec_type == 'inv':
            # delta @ Sigma^{-1} @ delta^T  per row
            quad      = (delta @ prec_data) * delta  # (N, d)
            distances = np.sqrt(np.maximum(quad.sum(axis=1), 0.0))  # (N,)

        else:  # 'diag'
            # prec_data = 1 / sigma_diag  (d,)
            quad      = delta ** 2 * prec_data[None, :]  # (N, d)
            distances = np.sqrt(quad.sum(axis=1))         # (N,)

        return distances.astype(np.float32)
