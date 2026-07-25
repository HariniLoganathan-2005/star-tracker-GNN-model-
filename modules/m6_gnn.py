"""
Module 6 — GNN Star Identification (Fallback)

Runs a Graph Neural Network to identify stars when triangle matching fails.
Each detected star's local neighbourhood forms a graph; the GNN produces
rotation-invariant embeddings that are matched against pre-computed
catalogue embeddings via nearest-neighbour search.

Requires: torch  (pip install torch)
"""

import os, sys, logging, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# ── optional torch import ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — GNN module disabled. "
                   "Install with: pip install torch")

# ── constants ─────────────────────────────────────────────────────────────────
GNN_K          = 6      # k-NN connectivity
GNN_NODE_DIM   = 6      # (mag_norm, d1, d2, d3, d4, d5)
GNN_HIDDEN     = 64
GNN_EMBED_DIM  = 128
GNN_LAYERS     = 3
GNN_RADIUS_DEG = 15.0   # neighbourhood radius for catalogue embeddings

GNN_MODEL_FILE  = os.path.join(config.MODELS_DIR, "star_gnn.pt")
GNN_CATDB_FILE  = os.path.join(config.MODELS_DIR, "gnn_catalogue_db.pkl")


# ── model definition ──────────────────────────────────────────────────────────
if TORCH_AVAILABLE:

    class _SAGEConv(nn.Module):
        """GraphSAGE-style message-passing layer (no external dependency)."""
        def __init__(self, ch):
            super().__init__()
            self.W_self  = nn.Linear(ch, ch, bias=False)
            self.W_neigh = nn.Linear(ch, ch, bias=False)
            self.bn      = nn.BatchNorm1d(ch)

        def forward(self, x, edge_index):
            src, dst = edge_index                              # [E], [E]
            agg = torch.zeros_like(x)
            cnt = torch.zeros(x.size(0), 1, device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(1)), x[src])
            cnt.scatter_add_(0, dst.unsqueeze(-1),
                             torch.ones(src.size(0), 1, device=x.device))
            agg = agg / cnt.clamp(min=1)
            return self.bn(F.relu(self.W_self(x) + self.W_neigh(agg)))

    class StarGNN(nn.Module):
        """
        Star-neighbourhood GNN.
        Input : node features  [N, node_dim]
        Output: L2-normalised embeddings [N, embed_dim]
        """
        def __init__(self, node_dim=GNN_NODE_DIM, hidden=GNN_HIDDEN,
                     embed_dim=GNN_EMBED_DIM, n_layers=GNN_LAYERS):
            super().__init__()
            self.proj  = nn.Linear(node_dim, hidden)
            self.convs = nn.ModuleList([_SAGEConv(hidden) for _ in range(n_layers)])
            self.head  = nn.Sequential(
                nn.Linear(hidden * (n_layers + 1), embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        def forward(self, x, edge_index):
            h  = F.relu(self.proj(x))
            hs = [h]
            for conv in self.convs:
                h = h + conv(h, edge_index)   # residual
                hs.append(h)
            return F.normalize(self.head(torch.cat(hs, dim=-1)), dim=-1)


# ── graph helpers ─────────────────────────────────────────────────────────────

def _knn_edges(positions, k=GNN_K):
    """Build a k-NN directed edge_index [2, E] from 2-D positions."""
    from scipy.spatial import KDTree
    n = len(positions)
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    tree   = KDTree(positions)
    _, idx = tree.query(positions, k=min(k + 1, n))
    src, dst = [], []
    for i, nbrs in enumerate(idx):
        for j in nbrs:
            if j != i:
                src.append(i); dst.append(j)
    return np.array([src, dst], dtype=np.int64)


def _node_features(positions_deg, mags):
    """Return float32 node-feature matrix [N, 6] (rotation invariant distances)."""
    pos = np.array(positions_deg, dtype=np.float32)
    m = np.array(mags, dtype=np.float32)
    m_norm = (m - m.min()) / 3.0
    
    n = len(pos)
    features = np.zeros((n, 6), dtype=np.float32)
    features[:, 0] = m_norm
    
    for i in range(n):
        dists = np.linalg.norm(pos - pos[i], axis=1) / GNN_RADIUS_DEG
        dists = np.sort(dists)
        n_dists = min(len(dists) - 1, 5)
        if n_dists > 0:
            features[i, 1:1+n_dists] = dists[1:1+n_dists]
        if n_dists < 5:
            features[i, 1+n_dists:] = 1.0
            
    return features


def _embed(positions_deg, mags, model):
    """Run the GNN on a star patch; returns np.ndarray [N, embed_dim] or None."""
    if len(positions_deg) < 2:
        return None
    x  = torch.tensor(_node_features(positions_deg, mags))
    ei = torch.tensor(_knn_edges(positions_deg), dtype=torch.long)
    with torch.no_grad():
        return model(x, ei).numpy()


# ── pixel → angular offsets ───────────────────────────────────────────────────

def _pixels_to_angular(detected_stars, camera):
    """Convert pixel (u,v) → (dRA, dDec) in degrees via pinhole model."""
    f = camera.focal_length
    cx, cy = camera.cx, camera.cy
    positions, mags = [], []
    for s in detected_stars:
        dra  =  np.degrees(np.arctan2(s.u - cx,  f))
        ddec = -np.degrees(np.arctan2(s.v - cy,  f))
        positions.append([dra, ddec])
        mags.append(s.instrumental_mag)
    return np.array(positions), np.array(mags)


# ── catalogue embedding helpers ───────────────────────────────────────────────

def _project_neighbours(ref_vec, nbr_vecs):
    """
    Project neighbouring unit vectors onto the local tangent plane of ref_vec.
    Returns (dRA, dDec) pairs in approximate degrees.
    """
    north = np.array([0.0, 0.0, 1.0]) - ref_vec[2] * ref_vec
    nn_len = np.linalg.norm(north)
    if nn_len < 1e-9:
        north = np.array([0.0, 1.0, 0.0]) - ref_vec[1] * ref_vec
        nn_len = np.linalg.norm(north)
    north /= nn_len
    east   = np.cross(north, ref_vec)
    east  /= np.linalg.norm(east) + 1e-9
    diffs  = nbr_vecs - ref_vec[None, :]
    return np.column_stack([
        np.degrees(diffs @ east),
        np.degrees(diffs @ north),
    ])


# ── public: build catalogue database ─────────────────────────────────────────

def build_catalogue_embeddings(catalogue,
                               model_path=GNN_MODEL_FILE,
                               output_path=GNN_CATDB_FILE,
                               radius_deg=GNN_RADIUS_DEG):
    """
    Pre-compute a GNN embedding for every catalogue star.
    Run once after training:  python train_gnn.py --build-catalogue
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required.")
    model = _load_model(model_path)

    vecs  = catalogue['unit_vectors']
    mags  = catalogue['vmag']
    hids  = catalogue['hip_ids']
    n     = len(hids)
    cos_r = np.cos(np.radians(radius_deg))

    all_emb, all_ci = [], []
    logger.info(f"Building catalogue embeddings for {n} stars …")

    for ci in range(n):
        dots    = vecs @ vecs[ci]
        nbr_idx = np.where(dots >= cos_r)[0]
        if len(nbr_idx) < 3:
            continue
        positions = _project_neighbours(vecs[ci], vecs[nbr_idx]).tolist()
        nbr_mags  = mags[nbr_idx].tolist()
        emb = _embed(positions, nbr_mags, model)
        if emb is None:
            continue
        # central star is the one with dot==1
        c_pos = np.where(nbr_idx == ci)[0]
        if len(c_pos):
            all_emb.append(emb[c_pos[0]])
            all_ci.append(ci)
        if (ci + 1) % 1000 == 0:
            logger.info(f"  {ci+1}/{n}")

    db = {
        'embeddings':  np.array(all_emb,  dtype=np.float32),
        'cat_indices': np.array(all_ci,   dtype=np.int32),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved catalogue DB: {len(all_emb)} embeddings → {output_path}")
    return db


# ── public: inference ─────────────────────────────────────────────────────────

def gnn_identify_stars(detected_stars, camera, catalogue, tri_db):
    """
    Identify detected stars using the GNN fallback.

    Returns
    -------
    matches : list[StarMatch]
    success : bool
    """
    from modules.m5_triangle_match import StarMatch, _verify_geometric_consistency
    from modules.m4_pixel_to_vector import convert_pixels_to_vectors

    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available.")
        return [], False

    try:
        model    = _load_model()
        cat_data = _load_catalogue_db()
    except FileNotFoundError as e:
        logger.error(str(e))
        return [], False

    positions_deg, mags = _pixels_to_angular(detected_stars, camera)
    
    det_emb_list = []
    for i in range(len(positions_deg)):
        pos_shifted = positions_deg - positions_deg[i]
        dists = np.linalg.norm(pos_shifted, axis=1)
        nbr_idx = np.where(dists <= GNN_RADIUS_DEG)[0]
        
        if len(nbr_idx) < 4:
            det_emb_list.append(np.zeros(GNN_EMBED_DIM, dtype=np.float32))
            continue
            
        pos_nbrs = pos_shifted[nbr_idx].tolist()
        mag_nbrs = mags[nbr_idx].tolist()
        
        emb_all = _embed(pos_nbrs, mag_nbrs, model)
        if emb_all is None:
            det_emb_list.append(np.zeros(GNN_EMBED_DIM, dtype=np.float32))
            continue
            
        c_pos = np.where(nbr_idx == i)[0][0]
        det_emb_list.append(emb_all[c_pos])
        
    det_emb = np.array(det_emb_list)

    cat_emb = cat_data['embeddings']   # [M, D]
    cat_ci  = cat_data['cat_indices']  # [M]

    from scipy.spatial import KDTree
    tree = KDTree(cat_emb)
    K = 10
    _, nn_idx = tree.query(det_emb, k=K)

    # 1. Gather candidates
    star_vectors = convert_pixels_to_vectors(detected_stars, camera)
    body_vecs = np.array([s.body_vector for s in star_vectors])
    n_det = len(detected_stars)
    
    # 2. Vote matrix: rows=detected_star, cols=K candidates
    votes = np.zeros((n_det, K))
    
    cat_vecs = catalogue['unit_vectors']
    
    # 3. Pairwise voting
    for i in range(n_det):
        for j in range(i + 1, n_det):
            dist_ij_rad = np.arccos(np.clip(np.dot(body_vecs[i], body_vecs[j]), -1.0, 1.0))
            dist_ij_deg = np.degrees(dist_ij_rad)
            
            # Check all candidate pairs
            for ki in range(K):
                ci = int(cat_ci[nn_idx[i, ki]])
                for kj in range(K):
                    cj = int(cat_ci[nn_idx[j, kj]])
                    if ci == cj: continue
                    
                    dist_cat_rad = np.arccos(np.clip(np.dot(cat_vecs[ci], cat_vecs[cj]), -1.0, 1.0))
                    dist_cat_deg = np.degrees(dist_cat_rad)
                    
                    if abs(dist_ij_deg - dist_cat_deg) < config.TRIANGLE_ANGLE_TOLERANCE_DEG:
                        votes[i, ki] += 1
                        votes[j, kj] += 1

    # 4. Resolve matches
    matches = []
    used_cat = set()
    for i in range(n_det):
        best_k = np.argmax(votes[i])
        if votes[i, best_k] >= 3:  # require at least 3 supporting distances
            ci = int(cat_ci[nn_idx[i, best_k]])
            if ci not in used_cat:
                matches.append(StarMatch(
                    star_index=i,
                    hip_id=int(catalogue['hip_ids'][ci]),
                    catalogue_index=ci,
                    confidence=float(votes[i, best_k]) / n_det,
                    vote_count=int(votes[i, best_k])
                ))
                used_cat.add(ci)

    success = len(matches) >= config.QUEST_MIN_MATCHES
    logger.info(f"GNN: {len(matches)} verified matches via voting (success={success})")
    return matches, success


# ── model / db I/O ────────────────────────────────────────────────────────────

def _load_model(path=GNN_MODEL_FILE):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed.")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GNN model not found: {path}\n"
            f"Train first:  python train_gnn.py")
    m = StarGNN()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


def _load_catalogue_db(path=GNN_CATDB_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Catalogue DB not found: {path}\n"
            f"Build first:  python train_gnn.py --build-catalogue")
    with open(path, 'rb') as f:
        return pickle.load(f)
