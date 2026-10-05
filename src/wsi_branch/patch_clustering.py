"""K-means clustering + representative patch selection -> WSI branch input sequence (Sec III-B)."""
import numpy as np
from sklearn.cluster import KMeans

from utils.logging_utils import get_logger

log = get_logger(__name__)


def cluster_and_select(patches, embeddings: np.ndarray, n_clusters: int = 20,
                        patches_per_cluster: int = 5, seed: int = 42):
    """
    patches: list of (x, y, PIL.Image), same order as embeddings rows.
    Returns: (selected_patches, cluster_labels) — selected_patches has length
    up to n_clusters * patches_per_cluster, forming the WSI branch input
    sequence {I1, ..., IN}.
    """
    n_samples = embeddings.shape[0]
    effective_k = min(n_clusters, n_samples)
    if effective_k < n_clusters:
        log.info(
            f"Only {n_samples} candidate patches available for {n_clusters} "
            f"clusters — reducing to {effective_k} clusters."
        )

    kmeans = KMeans(n_clusters=effective_k, n_init=20, random_state=seed)
    cluster_labels = kmeans.fit_predict(embeddings)

    rng = np.random.default_rng(seed)
    selected = []
    selected_indices = []
    for cluster_id in range(effective_k):
        idx = np.where(cluster_labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        n_sample = min(patches_per_cluster, len(idx))
        if n_sample < patches_per_cluster:
            log.info(f"Cluster {cluster_id} has only {len(idx)} patches — using all of them.")
        chosen = rng.choice(idx, size=n_sample, replace=False)
        for i in chosen:
            selected.append(patches[i])
            selected_indices.append(i)

    log.info(
        f"Selected {len(selected)} patches from {effective_k} clusters "
        f"(target: {n_clusters * patches_per_cluster})"
    )
    return selected, cluster_labels, np.array(selected_indices)