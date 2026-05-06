"""
SIFT: a TF-IDF + Random Forest baseline on byte histograms.

The features are 256-dim byte counts; we apply a TfidfTransformer (with
the standard ``smooth_idf=True`` formulation), L2-normalize, and feed the
result to a RandomForestClassifier. This deliberately matches the
``run_sift_standalone`` implementation we use in the paper so 4 KB / 8 KB
/ 16 KB histograms behave consistently with the 512 B SIFT.
"""

from typing import Tuple

import numpy as np
from scipy.sparse import diags as sparse_diags
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


def extract_byte_counts(blocks: np.ndarray, chunk_size: int = 500_000) -> np.ndarray:
    """Compute per-row 256-dim byte histograms.

    Args:
        blocks: ``(N, L)`` uint8 array of fragments.
        chunk_size: Number of fragments processed at once when going through
            ``np.bincount``; controls peak memory.

    Returns:
        ``(N, 256)`` float32 array of raw byte counts (no normalization).
    """
    n_samples, block_size = blocks.shape
    byte_counts = np.zeros((n_samples, 256), dtype=np.float32)

    # Bincount-based histogramming is much faster than per-row loops for the
    # sizes we care about (millions of rows).
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = blocks[start:end]
        n = end - start
        row_offsets = np.repeat(np.arange(n, dtype=np.int64), block_size) * 256
        flat_vals = chunk.ravel().astype(np.int64) + row_offsets
        counts = np.bincount(flat_vals, minlength=n * 256)
        byte_counts[start:end] = counts[: n * 256].reshape(n, 256)

    return byte_counts


def fit_tfidf(byte_counts: np.ndarray) -> Tuple[np.ndarray, TfidfTransformer]:
    """Fit a TF-IDF transformer on byte histograms; returns (features, transformer).

    The features are L2-normalized in place. The transformer is returned so
    we can apply it again at inference time.
    """
    n_docs = byte_counts.shape[0]
    df = np.count_nonzero(byte_counts, axis=0).astype(np.float64)
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    tfidf.idf_ = idf
    n_feat = len(idf)
    # sklearn expects ``_idf_diag`` to be a sparse diag; we set it manually so
    # the transformer is usable without re-fitting on a dummy corpus.
    tfidf._idf_diag = sparse_diags(
        idf, offsets=0, shape=(n_feat, n_feat), format="csr"
    )

    idf32 = idf.astype(np.float32)
    byte_counts *= idf32
    normalize(byte_counts, norm="l2", copy=False)
    return byte_counts, tfidf


def transform_with_tfidf(
    byte_counts: np.ndarray, tfidf: TfidfTransformer
) -> np.ndarray:
    """Apply a fitted TF-IDF + L2 normalize. Modifies ``byte_counts`` in place."""
    idf32 = tfidf.idf_.astype(np.float32)
    byte_counts *= idf32
    normalize(byte_counts, norm="l2", copy=False)
    return byte_counts


def train_sift_model(
    blocks: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 30,
    min_samples_leaf: int = 100,
    max_samples: float = None,
    n_jobs: int = -1,
    seed: int = 42,
    verbose: int = 1,
) -> dict:
    """Train a SIFT model end-to-end.

    Args:
        blocks: ``(N, L)`` uint8 fragments.
        labels: ``(N,)`` integer class labels.
        n_estimators / max_depth / min_samples_leaf: RandomForest hyperparams.
        max_samples: Optional bagging fraction (each tree sees this fraction
            of the training set).
        n_jobs: RandomForest parallelism. -1 means "use all cores".
        seed: RNG seed.
        verbose: Passed through to RandomForestClassifier.

    Returns:
        ``{'classifier': RandomForestClassifier, 'tfidf': TfidfTransformer}``.
    """
    counts = extract_byte_counts(blocks)
    features, tfidf = fit_tfidf(counts)

    rf_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=max(5, min_samples_leaf * 2),
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=seed,
    )
    if max_samples is not None:
        rf_kwargs["max_samples"] = max_samples

    clf = RandomForestClassifier(**rf_kwargs)
    clf.fit(features, labels)
    return {"classifier": clf, "tfidf": tfidf}


def predict_sift_model(
    bundle: dict,
    blocks: np.ndarray,
    return_top_k: int = 1,
) -> np.ndarray:
    """Predict class labels for a batch of fragments.

    Args:
        bundle: The dict returned by ``train_sift_model``.
        blocks: ``(N, L)`` uint8 fragments.
        return_top_k: If 1, returns ``(N,)`` top-1 predictions. If > 1,
            returns ``(N, k)`` top-k predictions sorted by descending probability.
    """
    counts = extract_byte_counts(blocks)
    features = transform_with_tfidf(counts, bundle["tfidf"])
    if return_top_k <= 1:
        return bundle["classifier"].predict(features)
    probs = bundle["classifier"].predict_proba(features)
    classes = bundle["classifier"].classes_
    top_idx = np.argsort(probs, axis=1)[:, -return_top_k:][:, ::-1]
    return classes[top_idx]
