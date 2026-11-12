"""
Unsupervised SAE feature clustering — group features by their relationships.

Computes pairwise feature similarity / interaction matrices from SAE codes on a
background corpus of model activations, then clusters features. No manifold
labels are used.

Similarity matrices (all ``[F, F]``):
  1. cosine        — decoder weight cosine similarity
  2. coactivation  — P(j fires | i fires), binary conditional co-occurrence
  3. correlation   — Pearson correlation of activation magnitudes
  4. mi            — mutual information (binned activations, normalized)
  5. ising         — inverse Ising couplings via L1 pseudolikelihood (LASSO)

Clustering:
  - Leiden community detection on a thresholded similarity graph
  - Spectral clustering on the affinity matrix

Workflow:
  # 1. Extract background activations (uses nnsight + a streaming text dataset).
  uv run background.py --n-tokens 500000

  # 2. Compute all five similarity matrices.
  uv run unsupervised_clustering.py matrices --sae /path/to/sae.pt --n-tokens 500000

  # 3. Cluster features (Ising recommended — see Sec. 5 of the paper).
  uv run unsupervised_clustering.py cluster --matrix ising --method leiden
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from data import D_MODEL, DEVICE, CACHE_DIR
from saes import load_sae, encode_sae, get_decoder
from background import load_background

MATRIX_DIR = CACHE_DIR / "matrices"
CLUSTER_DIR = CACHE_DIR / "clusters"

ALL_MATRICES = ['cosine', 'coactivation', 'correlation', 'mi', 'ising']


def _sae_tag(sae_path):
    """Stable per-SAE tag (stem + short hash of the absolute path)."""
    p = Path(sae_path).resolve()
    h = hashlib.sha1(str(p).encode()).hexdigest()[:8]
    return f"{p.stem}_{h}"


def _matrix_dir(sae_path):
    """Directory for matrices belonging to a specific SAE checkpoint."""
    return MATRIX_DIR / _sae_tag(sae_path)


# ── Code loading ─────────────────────────────────────────────────────────────

def load_or_encode_codes(sae_path, n_tokens=500_000, sae_kwargs=None):
    """Load cached SAE codes, encoding from background activations if needed.

    Returns a ``[n_tokens, n_features]`` float32 memmap. The cache is keyed
    by both ``n_tokens`` and the SAE checkpoint path so different SAEs do
    not silently share cached codes.
    """
    d = _matrix_dir(sae_path)
    cache = d / f"codes_{n_tokens}.dat"
    shape_path = cache.with_suffix(".shape.json")
    if cache.exists() and shape_path.exists():
        shape = tuple(json.load(open(shape_path)))
        print(f"Loading cached codes (mmap): {cache} {shape}", flush=True)
        return np.memmap(cache, dtype=np.float32, mode='r', shape=shape)

    bg = load_background(n_tokens)
    sae = load_sae(sae_path, device=DEVICE, **(sae_kwargs or {}))

    cache.parent.mkdir(parents=True, exist_ok=True)
    N = bg.shape[0]
    chunk = 8192
    codes_mmap = None
    total = 0

    print(f"Encoding {N} background activations through SAE...", flush=True)
    for start in tqdm(range(0, N, chunk), desc="encode"):
        end = min(start + chunk, N)
        z = encode_sae(sae, torch.from_numpy(np.asarray(bg[start:end])))
        if codes_mmap is None:
            n_feat = z.shape[1]
            codes_mmap = np.memmap(cache, dtype=np.float32, mode='w+',
                                   shape=(N, n_feat))
        codes_mmap[start:end] = z
        total = end

    codes_mmap.flush()
    shape = (total, codes_mmap.shape[1])
    del codes_mmap
    with open(shape_path, 'w') as f:
        json.dump(shape, f)
    print(f"Wrote {shape} codes to {cache}", flush=True)
    return np.memmap(cache, dtype=np.float32, mode='r', shape=shape)


# ── 1. Cosine similarity (decoder weights) ──────────────────────────────────

def compute_cosine(sae):
    """Cosine similarity between all pairs of decoder directions ``[F, F]``."""
    W = get_decoder(sae)
    norms = np.linalg.norm(W, axis=1, keepdims=True).clip(1e-8)
    W_normed = W / norms
    return W_normed @ W_normed.T


# ── 2. Binary conditional co-activation ──────────────────────────────────────

def compute_coactivation(codes, chunk_size=50_000):
    """``coact[i, j] = P(j fires | i fires)`` for all feature pairs ``[F, F]``."""
    N, F = codes.shape
    joint = np.zeros((F, F), dtype=np.float64)
    marginal = np.zeros(F, dtype=np.float64)
    n_chunks = (N + chunk_size - 1) // chunk_size
    for ci, i in enumerate(range(0, N, chunk_size)):
        B_chunk = (codes[i:i + chunk_size] > 0).astype(np.float32)
        joint += B_chunk.T @ B_chunk
        marginal += B_chunk.sum(0)
        print(f"  coactivation chunk {ci+1}/{n_chunks}", flush=True)
    return (joint / marginal[:, None].clip(1)).astype(np.float32)


# ── 3. Activation correlation ────────────────────────────────────────────────

def compute_correlation(codes, active_only=True, chunk_size=50_000):
    """Pearson correlation of activation magnitudes ``[F, F]``.

    Only computes correlations for features that fire on >0.1% of samples.
    """
    N, F = codes.shape
    fr = np.zeros(F, dtype=np.float64)
    means = np.zeros(F, dtype=np.float64)
    for i in range(0, N, chunk_size):
        chunk = codes[i:i + chunk_size]
        fr += (chunk > 0).sum(0)
        means += chunk.sum(0)
    fr /= N
    means /= N

    active = (np.where(fr > 0.001)[0] if active_only else np.arange(F))

    corr = np.zeros((F, F), dtype=np.float32)
    if len(active) == 0:
        return corr

    active_means = means[active].astype(np.float32)
    n_chunks = (N + chunk_size - 1) // chunk_size
    sq_sum = np.zeros(len(active), dtype=np.float64)
    for ci, i in enumerate(range(0, N, chunk_size)):
        sub = codes[i:i + chunk_size][:, active].astype(np.float32) - active_means
        sq_sum += (sub ** 2).sum(0)
        print(f"  correlation stds chunk {ci+1}/{n_chunks}", flush=True)
    stds = np.sqrt(sq_sum / N).clip(1e-8).astype(np.float32)

    C = np.zeros((len(active), len(active)), dtype=np.float64)
    for ci, i in enumerate(range(0, N, chunk_size)):
        sub = codes[i:i + chunk_size][:, active].astype(np.float32)
        sub = (sub - active_means) / stds
        C += sub.T @ sub
        print(f"  correlation matmul chunk {ci+1}/{n_chunks}", flush=True)

    C = (C / N).astype(np.float32)
    corr[np.ix_(active, active)] = C
    return corr


# ── 4. Mutual information ───────────────────────────────────────────────────

def compute_mi(codes, n_bins=8, normalize='geometric', chunk_size=128):
    """Pairwise normalized mutual information ``[F, F]``.

    Bins each feature's activations into ``n_bins`` quantile bins and computes
    ``MI(X_i, X_j)`` on GPU. Normalized by geometric mean of marginal entropies
    so values lie in ``[0, 1]``.
    """
    N, F = codes.shape
    eps = 1e-12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_bins = n_bins + 1

    bins_np = np.zeros((N, F), dtype=np.int8)
    for ci in range(0, F, 1000):
        ef = min(ci + 1000, F)
        for f in range(ci, ef):
            col = codes[:, f]
            nonzero = col > 0
            if nonzero.sum() < 2:
                continue
            vals = col[nonzero]
            edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])
            bins_np[nonzero, f] = np.digitize(vals, edges).clip(0, n_bins - 1).astype(np.int8) + 1
        print(f"  Binned features {ci}-{ef}/{F}", flush=True)

    entropies = np.zeros(F, dtype=np.float32)
    for f in range(F):
        hist = np.bincount(bins_np[:, f].astype(np.int32), minlength=total_bins).astype(np.float32)
        p = hist / N
        entropies[f] = -(p * np.log2(p + eps)).sum()
    entropies_gpu = torch.from_numpy(entropies).to(device)

    mi_mat = np.zeros((F, F), dtype=np.float32)
    n_outer = (F + chunk_size - 1) // chunk_size
    total_pairs = n_outer * (n_outer + 1) // 2
    print(f"Computing MI ({device}): {F} features, {total_pairs} chunk-pairs",
          flush=True)
    t0 = time.time()
    done = 0
    for ci in range(0, F, chunk_size):
        ei = min(ci + chunk_size, F)
        bins_i = torch.from_numpy(bins_np[:, ci:ei].astype(np.int64)).to(device)
        onehot_i = torch.nn.functional.one_hot(bins_i, total_bins).float()
        px = onehot_i.mean(dim=0)

        for cj in range(ci, F, chunk_size):
            ej = min(cj + chunk_size, F)
            bins_j = torch.from_numpy(bins_np[:, cj:ej].astype(np.int64)).to(device)
            onehot_j = torch.nn.functional.one_hot(bins_j, total_bins).float()
            py = onehot_j.mean(dim=0)
            pxy = torch.einsum('nib,njc->ijbc', onehot_i, onehot_j) / N
            px_py = px.unsqueeze(1).unsqueeze(3) * py.unsqueeze(0).unsqueeze(2)
            mi_chunk = ((pxy + eps) * (torch.log2(pxy + eps)
                                       - torch.log2(px_py + eps))).sum(dim=(2, 3))
            if normalize == 'geometric':
                h_i = entropies_gpu[ci:ei].unsqueeze(1)
                h_j = entropies_gpu[cj:ej].unsqueeze(0)
                mi_chunk = torch.clamp(
                    mi_chunk / torch.sqrt(h_i * h_j + eps), 0.0, 1.0)
            mi_cpu = mi_chunk.cpu().numpy()
            mi_mat[ci:ei, cj:ej] = mi_cpu
            if ci != cj:
                mi_mat[cj:ej, ci:ei] = mi_cpu.T
            done += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if done % 100 == 0:
            print(f"  MI progress: {done}/{total_pairs}", flush=True)

    print(f"MI done in {time.time()-t0:.1f}s", flush=True)
    np.fill_diagonal(mi_mat, 1.0 if normalize else entropies)
    return mi_mat


# ── 5. Inverse Ising model ──────────────────────────────────────────────────

def _plm_init(csc, c_reg, max_it):
    global _csc, _C, _mi
    _csc = csc
    _C = c_reg
    _mi = max_it


def _plm_fit(i):
    from scipy.sparse import hstack as sp_hstack
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(_csc[:, i].todense()).ravel()
    if y.sum() == 0 or y.sum() == len(y):
        return i, None, None
    F = _csc.shape[1]
    if i == 0:
        X = _csc[:, 1:]
    elif i == F - 1:
        X = _csc[:, :-1]
    else:
        X = sp_hstack([_csc[:, :i], _csc[:, i + 1:]], format='csc')
    clf = LogisticRegression(
        penalty='l1', C=_C, solver='liblinear',
        max_iter=_mi, tol=1e-4, random_state=42)
    clf.fit(X, y)
    return i, clf.intercept_[0], clf.coef_[0]


def compute_ising(codes, alpha=0.01, n_jobs=16, max_iter=1000, max_samples=None,
                  target_features=None):
    """Fit inverse Ising model via pseudolikelihood maximization (CPU L1-LR).

    Returns ``(J, h, stats)`` in the ±1 sign convention, i.e. with
    ``P(s_i=+1 | s_{-i}) = sigmoid(2*h_i + 2*Σ_j J_ij s_j)`` where
    ``s_j ∈ {-1, +1}``. Internally we fit one L1-logistic regression per
    feature on the {0,1} firing indicators (pseudolikelihood) and convert
    the resulting parameters: ``J_ij = b_ij / 4`` and
    ``h_i = (a_i + (1/2)·Σ_j b_ij) / 2``.
    """
    from scipy.sparse import csc_matrix, vstack as sp_vstack
    N, F = codes.shape
    if max_samples and N > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, max_samples, replace=False)
        idx.sort()
    else:
        idx = np.arange(N)
    N = len(idx)

    chunk = 10_000
    parts = []
    for start in range(0, N, chunk):
        batch_idx = idx[start:start + chunk]
        block = codes[batch_idx]
        parts.append(csc_matrix((block > 0).astype(np.float32)))
    S = sp_vstack(parts, format='csc')
    del parts
    S.sort_indices()

    firing_rates = np.asarray(S.getnnz(axis=0)).ravel() / N
    active = np.where((firing_rates > 0) & (firing_rates < 1.0))[0]
    F_active = len(active)
    active_to_idx = {feat: i for i, feat in enumerate(active)}

    if target_features is not None:
        fit_indices = [active_to_idx[f] for f in target_features
                       if f in active_to_idx]
        print(f"Fitting PLM: {N} samples, {F_active} active features, "
              f"{len(fit_indices)}/{F_active} regressions, alpha={alpha}")
    else:
        fit_indices = list(range(F_active))
        print(f"Fitting PLM: {N} samples, {F_active} active features, "
              f"alpha={alpha}")

    S_csc = S[:, active]
    S_csc.sort_indices()
    del S
    C_reg = 1.0 / alpha

    import multiprocessing as mp
    t0 = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(n_jobs, initializer=_plm_init,
                  initargs=(S_csc, C_reg, max_iter)) as pool:
        results = list(tqdm(
            pool.imap(_plm_fit, fit_indices),
            total=len(fit_indices), desc="PLM fit"))

    J_active = np.zeros((F_active, F_active), dtype=np.float32)
    h_active = np.zeros(F_active, dtype=np.float32)
    n_failed = 0
    for i, intercept, coefs in results:
        if coefs is None:
            n_failed += 1
            continue
        J_active[i, :i] = coefs[:i] / 4
        J_active[i, i + 1:] = coefs[i:] / 4
        h_active[i] = (intercept + coefs.sum() / 2) / 2

    J_sym = (J_active + J_active.T) / 2.0
    J = np.zeros((F, F), dtype=np.float32)
    h = np.zeros(F, dtype=np.float32)
    J[np.ix_(active, active)] = J_sym
    h[active] = h_active

    elapsed = time.time() - t0
    nnz = int((np.abs(J_sym) > 1e-6).sum() // 2)
    stats = dict(n_samples=N, n_features=F, n_active=F_active,
                 n_fit=len(fit_indices), alpha=alpha, n_failed=n_failed,
                 elapsed_min=elapsed / 60, nnz_couplings=nnz,
                 mean_abs_J=float(np.abs(J_sym[np.triu_indices(F_active, k=1)]).mean()),
                 method='cpu_liblinear')
    print(f"PLM done in {elapsed/60:.1f} min, {nnz} non-zero couplings")
    return J, h, stats


# ── Matrix I/O ───────────────────────────────────────────────────────────────

def save_matrix(name, matrix, sae_path, extra=None):
    """Save a matrix under the SAE-specific cache subdirectory."""
    d = _matrix_dir(sae_path)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.npy"
    np.save(path, matrix)
    if extra:
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(extra, f, indent=2)
    print(f"  Saved {path} {matrix.shape}")
    return path


def load_matrix(name, sae_path):
    """Load a saved matrix for a given SAE checkpoint.

    ``name`` may be a bare matrix name (``cosine``, ``ising``, ...) or an
    explicit ``.npy`` path (in which case ``sae_path`` is ignored).
    """
    if name.endswith(".npy"):
        return np.load(name)
    path = _matrix_dir(sae_path) / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"No matrix at {path}. Run `matrices` first.")
    return np.load(path)


# ── Clustering ───────────────────────────────────────────────────────────────

def _resolve_threshold(matrix, threshold=None, percentile=None):
    """Resolve an absolute threshold, optionally from a percentile of |M|."""
    if percentile is not None:
        M = matrix.copy()
        np.fill_diagonal(M, 0)
        vals = np.abs(M[np.triu_indices(M.shape[0], k=1)])
        t = float(np.percentile(vals, percentile))
        print(f"  Percentile {percentile} -> threshold {t:.6f}")
        return t
    return threshold if threshold is not None else 0.005


def cluster_leiden(matrix, threshold=None, percentile=None,
                   resolution=1.0, min_size=2):
    """Leiden community detection on a thresholded similarity graph."""
    import igraph as ig
    import leidenalg

    threshold = _resolve_threshold(matrix, threshold, percentile)
    F = matrix.shape[0]
    M = matrix.copy()
    np.fill_diagonal(M, 0)

    edges, weights = [], []
    for i in range(F):
        for j in range(i + 1, F):
            if abs(M[i, j]) > threshold:
                edges.append((i, j))
                weights.append(abs(float(M[i, j])))

    print(f"Leiden: {F} nodes, {len(edges)} edges (threshold={threshold})")
    g = ig.Graph(n=F, edges=edges, directed=False)
    g.es['weight'] = weights
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights='weight', resolution_parameter=resolution,
        n_iterations=-1, seed=42)

    clusters = {}
    for cid, members in enumerate(partition):
        if len(members) >= min_size:
            clusters[cid] = sorted(members)
    modularity = partition.quality()
    sizes = sorted([len(c) for c in clusters.values()], reverse=True)
    print(f"  {len(clusters)} clusters, modularity={modularity:.4f}")
    print(f"  sizes: {sizes[:20]}")
    return clusters, modularity


def cluster_spectral(matrix, n_clusters=None, max_k=20, threshold=None,
                     percentile=None):
    """Spectral clustering with auto-k via silhouette score."""
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    threshold = _resolve_threshold(matrix, threshold, percentile)
    M = np.abs(matrix.copy())
    np.fill_diagonal(M, 0)
    M[M < threshold] = 0

    active = np.where(M.sum(1) > 0)[0]
    if len(active) < 4:
        print("  Too few active features for spectral clustering")
        return {}, 0.0
    A = M[np.ix_(active, active)]
    # silhouette_score(metric='precomputed') expects distances, A is an affinity.
    D = A.max() - A
    np.fill_diagonal(D, 0)

    if n_clusters is not None:
        labels = SpectralClustering(
            n_clusters, affinity='precomputed', random_state=42
        ).fit_predict(A)
        sil = silhouette_score(D, labels, metric='precomputed') \
            if len(set(labels)) > 1 else 0
    else:
        best = (None, 2, -1.0)
        for k in range(2, min(max_k + 1, len(active))):
            labels = SpectralClustering(
                k, affinity='precomputed', random_state=42
            ).fit_predict(A)
            s = silhouette_score(D, labels, metric='precomputed') \
                if len(set(labels)) > 1 else 0
            if s > best[2]:
                best = (labels, k, s)
        labels, n_clusters, sil = best

    clusters = {}
    for cid in range(labels.max() + 1):
        members = active[labels == cid].tolist()
        if len(members) >= 2:
            clusters[cid] = sorted(members)
    print(f"  Spectral: {len(clusters)} clusters (k={n_clusters}), sil={sil:.3f}")
    return clusters, sil


def save_clusters(name, clusters):
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    path = CLUSTER_DIR / f"{name}.json"
    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in clusters.items()}, f, indent=2)
    print(f"  Saved {path}")


# ── Heatmaps ─────────────────────────────────────────────────────────────────

def plot_heatmap(matrix, features, title, out_path,
                 cmap='PiYG', vmin=None, vmax=None, center=None,
                 log_scale=False):
    """Plot a pairwise heatmap for a feature subset."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = matrix[np.ix_(features, features)].copy()
    np.fill_diagonal(sub, 0)
    if log_scale:
        eps = 1e-3
        sub = np.log(1 + np.abs(sub) / eps) * np.sign(sub)

    if vmin is None and center is not None:
        vmax = max(0.1, np.abs(sub).max())
        vmin = -vmax
    elif vmin is None:
        vmin = 0
        vmax = max(0.1, sub.max())

    fig, ax = plt.subplots(figsize=(12, 10))
    annot = len(features) <= 20
    sns.heatmap(sub, annot=annot, fmt=".3f" if annot else "",
                cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                xticklabels=features, yticklabels=features, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_all_heatmaps(features, sae_path, out_dir=None):
    """Plot heatmaps of every computed similarity matrix for a feature subset."""
    out_dir = (Path(out_dir) if out_dir
               else CACHE_DIR / "heatmaps" / _sae_tag(sae_path))
    configs = [
        ('cosine',       'Decoder cosine similarity',     'PiYG',    None, None, 0),
        ('coactivation', 'P(j fires | i fires)',          'Greens',  0,    None, None),
        ('correlation',  'Activation correlation',        'PiYG',    None, None, 0),
        ('mi',           'Mutual information (norm.)',    'Greens',  0,    None, None),
        ('ising',        'J_ij Ising couplings (log scale)', 'RdBu_r', None, None, 0),
    ]
    for name, title, cmap, vmin, vmax, center in configs:
        try:
            M = load_matrix(name, sae_path)
        except FileNotFoundError:
            print(f"  Skipping {name} — not computed yet")
            continue
        plot_heatmap(M, features, title, out_dir / f"{name}.png",
                     cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                     log_scale=(name == 'ising'))


# ── Cluster decoder-direction reconstruction plot ────────────────────────────

def run_reconstruct(cluster_path, sae_path, n_tokens=500_000,
                    max_points=5000, out_dir=None, norm_std=3.0,
                    sae_kwargs=None):
    """For each cluster, PCA the decoder directions and overlay the activations
    of tokens where any cluster feature fires.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    with open(cluster_path) as f:
        clusters = {k: v for k, v in json.load(f).items()}

    sae = load_sae(sae_path, device=DEVICE, **(sae_kwargs or {}))
    decoder = get_decoder(sae)
    codes = load_or_encode_codes(sae_path, n_tokens, sae_kwargs=sae_kwargs)
    acts = load_background(n_tokens)

    out_dir = Path(out_dir) if out_dir else CACHE_DIR / "cluster_pca"
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_name = Path(cluster_path).stem

    skipped = 0
    for cid, feat_indices in tqdm(clusters.items(), desc="Cluster PCA"):
        feat_indices = np.array(feat_indices)
        W = decoder[feat_indices]

        norms = np.linalg.norm(W, axis=1)
        std_norm = norms.std()
        if std_norm > 0:
            keep = np.abs(norms - norms.mean()) <= norm_std * std_norm
        else:
            keep = np.ones(len(norms), dtype=bool)
        W_clean = W[keep]
        feat_clean = feat_indices[keep]
        if W_clean.shape[0] < 4:
            skipped += 1
            continue

        pca = PCA(n_components=3)
        feat_coords = pca.fit_transform(W_clean)
        var = pca.explained_variance_ratio_ * 100

        partial_codes = codes[:, feat_indices]
        token_mask = (partial_codes > 0).any(axis=1)
        active_acts = acts[token_mask]

        if active_acts.shape[0] > 0:
            act_norms = np.linalg.norm(active_acts, axis=1)
            act_std = act_norms.std()
            if act_std > 0:
                act_keep = np.abs(act_norms - act_norms.mean()) <= norm_std * act_std
                active_acts = active_acts[act_keep]
        if active_acts.shape[0] > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(active_acts.shape[0], max_points, replace=False)
            active_acts = active_acts[idx]
        tok_coords = (pca.transform(active_acts)
                      if active_acts.shape[0] > 0 else np.empty((0, 3)))

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        if tok_coords.shape[0] > 0:
            ax.scatter(tok_coords[:, 0], tok_coords[:, 1], tok_coords[:, 2],
                       s=1, alpha=0.15, c="silver",
                       label=f"tokens ({tok_coords.shape[0]})")
        ax.scatter(feat_coords[:, 0], feat_coords[:, 1], feat_coords[:, 2],
                   s=40, alpha=0.9, c="crimson", marker="x", linewidths=1.5,
                   label=f"features ({len(feat_clean)})")
        if len(feat_clean) <= 50:
            for i, fid in enumerate(feat_clean):
                ax.text(feat_coords[i, 0], feat_coords[i, 1], feat_coords[i, 2],
                        str(fid), fontsize=5, alpha=0.6)
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
        ax.set_zlabel(f"PC3 ({var[2]:.1f}%)")
        ax.set_title(f"Cluster {cid} — {len(feat_clean)} features, "
                     f"{int(token_mask.sum())} active tokens", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_{cid}_{cluster_name}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"Done. Saved to {out_dir}/ ({len(clusters) - skipped} plots, "
          f"{skipped} skipped)")


# ── Commands ─────────────────────────────────────────────────────────────────

def run_matrices(sae_path, n_tokens=500_000, matrices=None, sae_kwargs=None,
                 ising_alpha=0.01, ising_n_jobs=16, ising_max_samples=None,
                 ising_max_iter=1000):
    """Compute every requested similarity matrix and save each as ``cache/matrices/{name}.npy``.

    The Ising matrix is fit with an L1-regularized pseudolikelihood
    (LASSO logistic regression per feature). It is the slowest of the five
    metrics — use ``--ising-max-samples`` to subsample if it's too expensive.
    """
    matrices = matrices or ALL_MATRICES
    sae = load_sae(sae_path, device=DEVICE, **(sae_kwargs or {}))
    codes = None  # lazy: only encode background codes if a matrix needs them
    for name in matrices:
        print(f"\n--- {name} ---")
        if name == 'cosine':
            save_matrix('cosine', compute_cosine(sae), sae_path)
            continue
        if codes is None:
            codes = load_or_encode_codes(sae_path, n_tokens, sae_kwargs=sae_kwargs)
        if name == 'coactivation':
            save_matrix('coactivation', compute_coactivation(codes), sae_path)
        elif name == 'correlation':
            save_matrix('correlation', compute_correlation(codes), sae_path)
        elif name == 'mi':
            save_matrix('mi', compute_mi(codes), sae_path)
        elif name == 'ising':
            J, h, stats = compute_ising(
                codes, alpha=ising_alpha, n_jobs=ising_n_jobs,
                max_iter=ising_max_iter, max_samples=ising_max_samples)
            save_matrix('ising', J, sae_path, extra=stats)
            np.save(_matrix_dir(sae_path) / 'ising_fields.npy', h)
        else:
            print(f"  Unknown matrix: {name}")


def run_ising(sae_path, n_tokens=500_000, alpha=0.01, n_jobs=16,
              max_iter=1000, max_samples=None, features=None,
              sae_kwargs=None):
    """Fit the inverse Ising model with custom solver settings.

    Use this when you want to tune ``alpha`` (L1 strength) or fit the
    couplings on a subset of feature indices. For the default pipeline,
    ``unsupervised_clustering.py matrices`` already computes Ising.
    """
    codes = load_or_encode_codes(sae_path, n_tokens, sae_kwargs=sae_kwargs)
    J, h, stats = compute_ising(codes, alpha=alpha, n_jobs=n_jobs,
                                max_iter=max_iter, max_samples=max_samples,
                                target_features=features)
    if features is not None:
        J = J[np.ix_(features, features)]
        h = h[features]
    tag = '_subset' if features is not None else ''
    save_matrix(f'ising{tag}', J, sae_path, extra=stats)
    np.save(_matrix_dir(sae_path) / f'ising{tag}_fields.npy', h)


def run_cluster(matrix_name, sae_path, method='leiden', threshold=None,
                percentile=None, resolution=1.0, n_clusters=None):
    M = load_matrix(matrix_name, sae_path)
    if method == 'leiden':
        clusters, _ = cluster_leiden(M, threshold=threshold,
                                     percentile=percentile,
                                     resolution=resolution)
    else:
        clusters, _ = cluster_spectral(M, n_clusters=n_clusters,
                                       threshold=threshold,
                                       percentile=percentile)
    save_clusters(f"{_sae_tag(sae_path)}_{method}_{Path(matrix_name).stem}",
                  clusters)
    return clusters


def run_heatmap(features, sae_path, out_dir=None):
    plot_all_heatmaps(features, sae_path, out_dir)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _add_sae_args(parser):
    parser.add_argument('--sae', type=str, required=True)
    parser.add_argument('--d-in', type=int, default=D_MODEL)
    parser.add_argument('--d-sae', type=int, default=None)
    parser.add_argument('--expansion-factor', type=int, default=None)
    parser.add_argument('--k', type=int, default=None)


def _sae_kwargs(a):
    return dict(d_in=a.d_in, d_sae=a.d_sae,
                expansion_factor=a.expansion_factor, k=a.k)


def main():
    p = argparse.ArgumentParser(
        description="Unsupervised SAE feature clustering")
    sp = p.add_subparsers(dest='cmd')

    mt = sp.add_parser('matrices',
                       help=f'Compute similarity matrices (default: all of {ALL_MATRICES})')
    _add_sae_args(mt)
    mt.add_argument('--n-tokens', type=int, default=500_000)
    mt.add_argument('--matrices', nargs='*', default=None,
                    help='Subset to compute; omit to compute all.')
    mt.add_argument('--ising-alpha', type=float, default=0.01,
                    help='L1 regularization for Ising pseudolikelihood')
    mt.add_argument('--ising-n-jobs', type=int, default=16,
                    help='Parallel workers for Ising PLM fits')
    mt.add_argument('--ising-max-samples', type=int, default=None,
                    help='Subsample background tokens before Ising fit')
    mt.add_argument('--ising-max-iter', type=int, default=1000,
                    help='Max liblinear iterations per Ising regression')

    ig = sp.add_parser('ising',
                       help='Refit Ising couplings with custom alpha or on a feature subset')
    _add_sae_args(ig)
    ig.add_argument('--n-tokens', type=int, default=500_000)
    ig.add_argument('--alpha', type=float, default=0.01,
                    help='L1 regularization for pseudolikelihood')
    ig.add_argument('--n-jobs', type=int, default=16)
    ig.add_argument('--max-iter', type=int, default=1000)
    ig.add_argument('--max-samples', type=int, default=None)
    ig.add_argument('--features', type=int, nargs='+', default=None,
                    help='Fit couplings only on this subset of feature indices')

    cl = sp.add_parser('cluster', help='Cluster features from a matrix')
    cl.add_argument('--sae', type=str, required=True,
                    help='Path to SAE checkpoint (identifies the matrix cache)')
    cl.add_argument('--matrix', type=str, required=True,
                    help='Matrix name (cosine, coactivation, correlation, mi, ising)')
    cl.add_argument('--method', choices=['leiden', 'spectral'], default='leiden')
    cl.add_argument('--threshold', type=float, default=None)
    cl.add_argument('--percentile', type=float, default=None,
                    help='Keep edges above this percentile of |off-diag|')
    cl.add_argument('--resolution', type=float, default=1.0)
    cl.add_argument('--n-clusters', type=int, default=None)

    rc = sp.add_parser('reconstruct',
                       help='PCA decoder directions + project activations per cluster')
    _add_sae_args(rc)
    rc.add_argument('--clusters', type=str, required=True)
    rc.add_argument('--n-tokens', type=int, default=500_000)
    rc.add_argument('--max-points', type=int, default=5000)
    rc.add_argument('--norm-std', type=float, default=3.0)
    rc.add_argument('--out-dir', type=str, default=None)

    hm = sp.add_parser('heatmap', help='Heatmap of a feature subset')
    hm.add_argument('--sae', type=str, required=True,
                    help='Path to SAE checkpoint (identifies the matrix cache)')
    hm.add_argument('--features', type=int, nargs='+', required=True)
    hm.add_argument('--out-dir', type=str, default=None)

    a = p.parse_args()
    if a.cmd == 'matrices':
        run_matrices(a.sae, a.n_tokens, a.matrices, sae_kwargs=_sae_kwargs(a),
                     ising_alpha=a.ising_alpha, ising_n_jobs=a.ising_n_jobs,
                     ising_max_samples=a.ising_max_samples,
                     ising_max_iter=a.ising_max_iter)
    elif a.cmd == 'ising':
        run_ising(a.sae, a.n_tokens, a.alpha, a.n_jobs, a.max_iter,
                  a.max_samples, a.features, sae_kwargs=_sae_kwargs(a))
    elif a.cmd == 'cluster':
        run_cluster(a.matrix, a.sae, method=a.method, threshold=a.threshold,
                    percentile=a.percentile, resolution=a.resolution,
                    n_clusters=a.n_clusters)
    elif a.cmd == 'reconstruct':
        run_reconstruct(a.clusters, a.sae, n_tokens=a.n_tokens,
                        max_points=a.max_points, out_dir=a.out_dir,
                        norm_std=a.norm_std, sae_kwargs=_sae_kwargs(a))
    elif a.cmd == 'heatmap':
        run_heatmap(a.features, a.sae, a.out_dir)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
