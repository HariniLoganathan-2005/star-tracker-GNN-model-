"""
GNN Training Script
===================
Trains the StarGNN model on synthetic star-field images and (optionally)
pre-computes catalogue embeddings.

Usage
-----
  # 1. Train the GNN
  python train_gnn.py

  # 2. Build catalogue embedding database (run after training)
  python train_gnn.py --build-catalogue

  # 3. Both in one go
  python train_gnn.py --build-catalogue --epochs 100
"""

import os, sys, argparse, logging, time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("train_gnn")

import config
from modules.m6_gnn import (
    StarGNN, _knn_edges, _node_features, _embed,
    GNN_MODEL_FILE, GNN_CATDB_FILE,
    TORCH_AVAILABLE, build_catalogue_embeddings,
    GNN_RADIUS_DEG
)

if not TORCH_AVAILABLE:
    logger.error("PyTorch is not installed. Run:  pip install torch")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_training_batch(catalogue, batch_size=128):
    """
    Generate a batch of star-centric patches directly from the catalogue.
    """
    from modules.m6_gnn import _project_neighbours, GNN_RADIUS_DEG
    vecs = catalogue['unit_vectors']
    mags = catalogue['vmag']
    n = len(mags)
    cos_r = np.cos(np.radians(GNN_RADIUS_DEG))
    
    samples = []
    indices = np.random.choice(n, batch_size)
    for ci in indices:
        dots = vecs @ vecs[ci]
        nbr_idx = np.where(dots >= cos_r)[0]
        if len(nbr_idx) < 4:
            continue
            
        positions = _project_neighbours(vecs[ci], vecs[nbr_idx]).tolist()
        nbr_mags  = mags[nbr_idx].tolist()
        
        # find index of the central star
        c_pos = np.where(nbr_idx == ci)[0][0]
        samples.append((positions, nbr_mags, c_pos))
        
    return samples


# ── NT-Xent contrastive loss ──────────────────────────────────────────────────

def nt_xent_loss(emb_a, emb_b, temperature=0.07):
    """
    Contrastive loss between two embedding sets.
    emb_a[i] and emb_b[i] are positive pairs.
    """
    n = emb_a.size(0)
    if n == 0:
        return torch.tensor(0.0, requires_grad=True)
    z = F.normalize(torch.cat([emb_a, emb_b], dim=0), dim=-1)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(-1e9)
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, labels.to(sim.device))


# ── Build augmented pair from one sample ─────────────────────────────────────

def _augment(positions, mags, missing_prob=0.1, noise_deg=0.05):
    """Add position/mag noise and randomly drop stars (but keep central)."""
    n = len(positions)
    pos = np.array(positions)
    mag = np.array(mags)
    
    # View A
    mask_a = np.random.rand(n) > missing_prob
    mask_a[0] = True # ensure central star (we will swap it to 0 later if needed, but let's just pass the indices)
    pos_a = pos + np.random.randn(n, 2) * noise_deg
    mag_a = mag + np.random.randn(n) * 0.1
    
    # View B
    mask_b = np.random.rand(n) > missing_prob
    mask_b[0] = True 
    pos_b = pos + np.random.randn(n, 2) * noise_deg
    mag_b = mag + np.random.randn(n) * 0.1
    
    return pos_a.tolist(), mag_a.tolist(), pos_b.tolist(), mag_b.tolist()


def _sample_to_tensors(positions, mags):
    x  = torch.tensor(_node_features(positions, mags))
    ei = torch.tensor(_knn_edges(positions), dtype=torch.long)
    return x, ei


# ── Training loop ─────────────────────────────────────────────────────────────

def train(catalogue, n_epochs=20, batch_size=128, lr=1e-3,
          model_path=GNN_MODEL_FILE):

    model     = StarGNN()
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs, eta_min=1e-5)

    logger.info(f"Training StarGNN for {n_epochs} epochs …")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        samples = generate_training_batch(catalogue, batch_size=batch_size)

        if not samples:
            continue

        model.train()
        
        batch_emb_a, batch_emb_b = [], []
        
        for positions, mags, c_pos in samples:
            # swap central star to index 0 for easy tracking
            positions[0], positions[c_pos] = positions[c_pos], positions[0]
            mags[0], mags[c_pos] = mags[c_pos], mags[0]
            
            pos_a, mag_a, pos_b, mag_b = _augment(positions, mags)

            xa, eia = _sample_to_tensors(pos_a, mag_a)
            xb, eib = _sample_to_tensors(pos_b, mag_b)

            emb_a = model(xa, eia)
            emb_b = model(xb, eib)
            
            batch_emb_a.append(emb_a[0]) # central star embedding
            batch_emb_b.append(emb_b[0])

        if not batch_emb_a:
            continue
            
        ea = torch.stack(batch_emb_a)
        eb = torch.stack(batch_emb_b)
        
        loss = nt_xent_loss(ea, eb)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch}/{n_epochs} | loss={loss.item():.4f} | time={elapsed:.1f}s")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), model_path)
            logger.info(f"  ✓ Saved best model (loss={best_loss:.4f})")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Model saved: {model_path}")
    return model


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train StarGNN")
    parser.add_argument('--epochs',          type=int,   default=80)
    parser.add_argument('--batch-size',      type=int,   default=16,
                        help="Synthetic images per epoch")
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--build-catalogue', action='store_true',
                        help="Also build catalogue embedding DB after training")
    parser.add_argument('--catalogue-only',  action='store_true',
                        help="Skip training; only build catalogue embeddings")
    args = parser.parse_args()

    from catalogue.hipparcos import load_catalogue
    logger.info("Loading catalogue …")
    catalogue = load_catalogue(config.CATALOGUE_FILE)
    logger.info(f"  {len(catalogue['hip_ids'])} stars loaded")

    if not args.catalogue_only:
        train(catalogue, n_epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr)

    if args.build_catalogue or args.catalogue_only:
        logger.info("Building catalogue embedding database …")
        build_catalogue_embeddings(catalogue)
        logger.info("Done.")


if __name__ == "__main__":
    main()
