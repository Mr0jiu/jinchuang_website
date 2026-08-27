"""
向量检索模块：FAISS 索引封装，支持 Flat（暴力检索）和 IVF（近似检索）
"""
import faiss
import numpy as np
import pickle
import os


class SimilaritySearch:
    def __init__(self, embedding_dim=768, index_path="checkpoints/faiss_index.bin",
                 index_type="flat", nlist=100):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index_type = index_type
        self.nlist = nlist
        self.metadata = []
        self._index_trained = False

        # 初始化索引
        if index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist,
                                            faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

    def train(self, embeddings: np.ndarray):
        """训练 IVF 索引的聚类中心（仅 IVF 类型需要）

        Args:
            embeddings: np.array [N, D]，训练样本数应 >= nlist
        """
        if self.index_type != "ivf":
            return

        if len(embeddings) < self.nlist:
            print(f"  警告: 训练样本数({len(embeddings)}) < nlist({self.nlist})，"
                  f"调整为 nlist={max(1, len(embeddings) // 2)}")
            self.nlist = max(1, len(embeddings) // 2)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist,
                                            faiss.METRIC_INNER_PRODUCT)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        print(f"  训练 IVF 索引 (nlist={self.nlist}, 样本数: {len(embeddings)})...")
        self.index.train(embeddings)
        self._index_trained = True

    def add_embeddings(self, embeddings, metadata_list):
        if len(embeddings) == 0:
            return
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding, top_k=5):
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # IVF 索引查询时调整 nprobe（提高查询精度）
        if self.index_type == "ivf":
            nprobe = min(self.nlist, max(1, top_k * 2))
            self.index.nprobe = nprobe

        similarities, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        results = []
        for score, idx in zip(similarities[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            results.append({
                "score": float(score),
                "metadata": self.metadata[idx]
            })
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        meta_path = self.index_path.replace(".bin", "_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "index_type": self.index_type,
                "nlist": self.nlist,
            }, f)
        print(f"索引已保存至 {self.index_path} ({self.index_type})")

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            meta_path = self.index_path.replace(".bin", "_meta.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    self.metadata = data.get("metadata", [])
                    self.index_type = data.get("index_type", "flat")
                    self.nlist = data.get("nlist", 100)
                else:
                    # 兼容旧版本（纯 list 格式）
                    self.metadata = data
                    self.index_type = "flat"
            else:
                # 兼容旧版元数据格式（纯 list）
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                self.index_type = "flat"
            print(f"索引已加载，共 {self.index.ntotal} 条记录 ({self.index_type})")
            # IVF 索引标记为已训练
            if self.index_type == "ivf" and hasattr(self.index, "is_trained"):
                self._index_trained = self.index.is_trained
            return True
        return False


# 快速测试
if __name__ == "__main__":
    # 测试 Flat
    searcher = SimilaritySearch(index_type="flat")
    fake_embs = np.random.randn(10, 512).astype(np.float32)
    fake_embs = fake_embs / np.linalg.norm(fake_embs, axis=1, keepdims=True)
    searcher.add_embeddings(fake_embs, [{"id": i} for i in range(10)])
    results = searcher.search(fake_embs[0], top_k=3)
    print(f"Flat 检索测试通过: {len(results)} 条结果")

    # 测试 IVF
    searcher2 = SimilaritySearch(index_type="ivf", nlist=3)
    searcher2.train(fake_embs)
    searcher2.add_embeddings(fake_embs, [{"id": i} for i in range(10)])
    results2 = searcher2.search(fake_embs[0], top_k=3)
    print(f"IVF 检索测试通过: {len(results2)} 条结果")
