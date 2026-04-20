"""
Drop-in replacements for sklearn.cluster.DBSCAN and KMeans.
Uses only numpy + scipy (both import fine on this Windows env).
sklearn and open3d both hang on import due to threading issues.

DBSCAN uses sparse_distance_matrix for O(n) BFS instead of Python-list BFS.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix


class DBSCAN:
    """Fast DBSCAN using scipy cKDTree sparse graph + numpy BFS."""
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        if n == 0:
            self.labels_ = np.array([], dtype=np.intp)
            return self

        tree = cKDTree(X)
        # Build sparse adjacency — much faster than query_ball_point for BFS
        sparse_dist = tree.sparse_distance_matrix(tree, max_distance=self.eps, output_type='coo_matrix')
        
        # Convert to CSR for fast row slicing
        adj = sparse_dist.tocsr()
        
        # Count neighbors per point (including self)
        neighbor_counts = np.diff(adj.indptr)
        is_core = neighbor_counts >= self.min_samples
        
        labels = np.full(n, -1, dtype=np.intp)
        cluster_id = 0

        for seed in range(n):
            if labels[seed] != -1 or not is_core[seed]:
                continue
            # BFS using numpy array as queue
            labels[seed] = cluster_id
            queue = adj.indices[adj.indptr[seed]:adj.indptr[seed+1]].tolist()
            head = 0
            while head < len(queue):
                q = queue[head]
                head += 1
                if labels[q] == cluster_id:
                    continue
                if labels[q] == -1 or labels[q] == -1:
                    labels[q] = cluster_id
                if is_core[q]:
                    nbrs = adj.indices[adj.indptr[q]:adj.indptr[q+1]]
                    for nb in nbrs:
                        if labels[nb] == -1:
                            queue.append(nb)
            cluster_id += 1

        self.labels_ = labels
        return self


class KMeans:
    """Simple KMeans — only n_clusters=2 needed for lane split."""
    
    def __init__(self, n_clusters=2, random_state=0, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.labels_ = None
        self.cluster_centers_ = None
    
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if len(X) == 0:
            return self
        
        rng = np.random.RandomState(self.random_state)
        best_inertia = np.inf
        best_labels = None
        best_centers = None
        
        for _ in range(self.n_init):
            idx = rng.choice(len(X), size=self.n_clusters, replace=False)
            centers = X[idx].copy()
            
            for _iter in range(100):
                dists = np.stack([np.linalg.norm(X - c, axis=1) for c in centers], axis=1)
                labels = dists.argmin(axis=1)
                new_centers = np.array([X[labels == k].mean(axis=0) if (labels == k).any() 
                                        else centers[k] for k in range(self.n_clusters)])
                if np.allclose(new_centers, centers, atol=1e-8):
                    break
                centers = new_centers
            
            inertia = sum(np.linalg.norm(X[labels == k] - centers[k], axis=1).sum() 
                         for k in range(self.n_clusters))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers
        
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        return self
