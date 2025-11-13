import faiss
import numpy as np
from HSmodule import VectorRecord, DSU

class FaissSearch:
    def __init__(self):
        self.threshold = 0
        self.index = None
        self.metric = None # "cosine" hoặc "hamming"
        self.dim = 384
        self.bbit = 8

    def setDisFunc(self, metric: str):
        self.metric = metric
        if (metric == "cosine"):
            # Tạo 1 đồ thị HNSW trong không gian 384 chiều, mỗi node có tối đa 32 cạnh
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT 
            self.index.hnsw.efSearch = 64 #Số lượng node được duyệt khi tìm kiếm
        elif metric == "hamming":
            # Với b-bit minhash → binary vector, tổng số bit = dim * bbit
            n_bits = self.dim * self.bbit
            self.index = faiss.IndexBinaryFlat(n_bits)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
    def cosineSimilarity(self, vecs: np.ndarray, k: int): 
        vecs = vecs.astype("float32")
        faiss.normalize_L2(vecs) 
        self.index.reset()
        self.index.add(vecs)
        k = min(k, len(vecs))
        sims, idxs = self.index.search(vecs, k)
        sims = (sims - sims.min()) / (sims.max() - sims.min())
        return sims, idxs
    
    def hammingDistance(self, vecs: list[np.ndarray], k: int):
        # Chỉ lấy 8 bit cuối mỗi số, chuyển trực tiếp thành uint8
        bin_array = np.stack([np.array([x % 256 for x in v], dtype=np.uint8) for v in vecs])

        if self.index is None:
            self.index = faiss.IndexBinaryFlat(bin_array.shape[1]*8)

        self.index.reset()
        self.index.add(bin_array)
        k = min(k, len(vecs))
        dists, idxs = self.index.search(bin_array, k)
        return dists, idxs

    def classify(self, setOfVecRecord : list[VectorRecord]) -> list[list[VectorRecord]]: 
        if not setOfVecRecord:
            return []
        
        n = len(setOfVecRecord)
        self.dim = len(setOfVecRecord[0].vec)

        self.setDisFunc(self.metric)

        if self.metric is None:
            raise ValueError("Metric not set. Call setDisFunc(metric) before classify().")

        if self.metric == "cosine":
            vecs = np.stack([v.vec for v in setOfVecRecord])
            sims, idxs = self.cosineSimilarity(vecs, k=100)

        elif self.metric == "hamming":
            vecs = [v.vec for v in setOfVecRecord]
            dists, idxs = self.hammingDistance(vecs, k=100)

        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        sims = 1 - sims  # Chuyển similarity thành distance

        dsu = DSU(n)
        for i in range(n):
            if self.metric == "cosine":
                for j, sim in zip(idxs[i], sims[i]):
                    if i == j or j == -1:
                        continue
                    if sim <= self.threshold:
                        dsu.unionSet(i, j)
            elif self.metric == "hamming":
                for j, dist in zip(idxs[i], dists[i]):
                    if i == j or j == -1:
                        continue
                    if dist <= self.threshold:
                        dsu.unionSet(i, j)

        groups_idx = dsu.getGroups()
        groups = [[setOfVecRecord[i] for i in group] for group in groups_idx if group]
        return groups


        
