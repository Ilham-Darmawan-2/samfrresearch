import os
import time
import logging
import numpy as np
from pymilvus import MilvusClient, MilvusException


class MilvusDB:
    """
    Wrapper class untuk mengelola database Milvus berbasis file (.db)
    dengan fitur CRUD, search, auto reconnect, dan logging terintegrasi.
    """

    def __init__(
        self,
        db_path: str,
        collection_name: str = "face_embeddings",
        dim: int = 512,
        reconnect_attempts: int = 3,
        reconnect_delay: int = 2,
        log_file: str = "milvus.log",
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.dim = dim
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.client = None

        self._setup_logger(log_file)
        self._connect()
        self._ensure_collection_exists()

    # ==================================================
    # ================ LOGGING SYSTEM ==================
    # ==================================================
    def _setup_logger(self, log_file: str):
        self.logger = logging.getLogger("MilvusDB")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # ==================================================
    # ================ CONNECTION ======================
    # ==================================================
    def _connect(self):
        for attempt in range(self.reconnect_attempts):
            try:
                db_dir = os.path.dirname(self.db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir)

                self.client = MilvusClient(self.db_path)
                _ = self.client.list_collections()
                self.logger.info(f"Terhubung ke Milvus DB: {self.db_path}")
                return
            except Exception as e:
                self.logger.warning(
                    f"Gagal koneksi (percobaan {attempt+1}/{self.reconnect_attempts}): {e}"
                )
                time.sleep(self.reconnect_delay)

        self.logger.error("Gagal terhubung ke Milvus setelah beberapa percobaan.")
        raise ConnectionError("Tidak bisa terhubung ke Milvus DB")

    def _ensure_connection_alive(self):
        try:
            _ = self.client.list_collections()
        except Exception:
            self.logger.warning("Koneksi Milvus terputus, mencoba reconnect...")
            self._connect()

    # ==================================================
    # ================ COLLECTION HANDLER ==============
    # ==================================================
    def _ensure_collection_exists(self):
        try:
            if not self.client.has_collection(self.collection_name):
                self.logger.info(f"Membuat collection baru: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.dim,
                    metric_type="COSINE",
                    auto_id=False,
                )
            else:
                schema = self.client.describe_collection(self.collection_name)
                existing_dim = schema.get("dimension", None)
                if existing_dim and existing_dim != self.dim:
                    self.logger.warning(
                        f"Collection '{self.collection_name}' sudah ada dengan dimensi {existing_dim}, "
                        f"sedangkan inisialisasi dim={self.dim}. "
                        f"Gunakan collection lain atau hapus collection lama."
                    )
        except MilvusException as e:
            self.logger.error(f"Gagal memastikan collection: {e}")
            raise

    # ==================================================
    # ================ CRUD OPERATIONS =================
    # ==================================================
    def insert(self, embedding: np.ndarray, metadata: dict = None, similiarScoreThreshold: float = 0.98) -> int | None:
        try:
            self._ensure_connection_alive()
            
            if metadata is None:
                metadata = {}

            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            res = self.search(embedding, top_k=1)

            if embedding.shape[1] != self.dim:
                msg = (
                    f"Dimensi embedding ({embedding.shape[1]}) tidak cocok dengan collection ({self.dim}). "
                    f"Gunakan DB baru dengan dim={embedding.shape[1]}.\n"
                    f"Contoh: MilvusDB('{self.db_path}', collection_name='face_embeddings_{embedding.shape[1]}', dim={embedding.shape[1]})"
                )
                self.logger.error(msg)
                print(f"[ERROR] {msg}")
                return None

            if len(res) == 0 or res[0]["score"] < similiarScoreThreshold:
                record_id = int(time.time() * 1e6)
                data = {"id": record_id, "vector": embedding[0].tolist()}
                data.update(metadata)

                self.client.insert(collection_name=self.collection_name, data=[data])
                self.logger.info(f"Data inserted ID={record_id}, metadata={metadata}")
                print(f"[INFO] Data inserted ID={record_id}, metadata={metadata}")
            else:
                print(f"[INFO] Data already in database")

        except Exception as e:
            self.logger.error(f"Insert gagal: {e}")
            print(f"[ERROR] Insert gagal: {e}")

    def get(self, record_id: int) -> dict | None:
        try:
            self._ensure_connection_alive()
            result = self.client.query(
                collection_name=self.collection_name,
                filter=f"id == {record_id}",
                output_fields=["*"],  # ambil semua field termasuk metadata
            )
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Get gagal: {e}")
            print(f"[ERROR] Get gagal: {e}")
            return None

    def update(self, record_id: int, new_embedding: np.ndarray = None, new_metadata: dict = None) -> bool:
        try:
            existing = self.get(record_id)
            if not existing:
                self.logger.warning(f"Data ID={record_id} tidak ditemukan untuk update.")
                print(f"[WARN] Data ID={record_id} tidak ditemukan untuk update.")
                return False

            if new_embedding is not None:
                if new_embedding.ndim == 1:
                    new_embedding = new_embedding.reshape(1, -1)
                if new_embedding.shape[1] != self.dim:
                    msg = (
                        f"Dimensi embedding update ({new_embedding.shape[1]}) tidak cocok dengan collection ({self.dim}). "
                        f"Buat DB baru dengan dim={new_embedding.shape[1]}."
                    )
                    self.logger.error(msg)
                    print(f"[ERROR] {msg}")
                    return False
                existing["vector"] = new_embedding[0].tolist()

            if new_metadata:
                existing.update(new_metadata)

            self.client.upsert(collection_name=self.collection_name, data=[existing])
            self.logger.info(f"Data ID={record_id} berhasil diupdate.")
            print(f"[INFO] Data ID={record_id} berhasil diupdate.")
            return True
        except Exception as e:
            self.logger.error(f"Update gagal: {e}")
            print(f"[ERROR] Update gagal: {e}")
            return False

    def delete(self, record_id: int) -> bool:
        try:
            self._ensure_connection_alive()
            self.client.delete(
                collection_name=self.collection_name,
                filter=f"id == {record_id}",
            )
            self.logger.info(f"Data ID={record_id} dihapus.")
            print(f"[INFO] Data ID={record_id} dihapus.")
            return True
        except Exception as e:
            self.logger.error(f"Delete gagal: {e}")
            print(f"[ERROR] Delete gagal: {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        try:
            self._ensure_connection_alive()
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            if query_embedding.shape[1] != self.dim:
                msg = (
                    f"Dimensi query ({query_embedding.shape[1]}) tidak cocok dengan collection ({self.dim}). "
                    f"Gunakan DB baru dengan dim={query_embedding.shape[1]}."
                )
                self.logger.error(msg)
                print(f"[ERROR] {msg}")
                return []

            results = self.client.search(
                collection_name=self.collection_name,
                data=query_embedding,
                limit=top_k,
                output_fields=["*"],  # ambil semua field termasuk metadata
            )

            formatted = []
            for hit in results[0]:
                info = hit["entity"]
                info["score"] = hit["distance"]
                formatted.append(info)

            self.logger.info(f"Search berhasil, ditemukan {len(formatted)} hasil.")
            return formatted

        except Exception as e:
            self.logger.error(f"Search gagal: {e}")
            print(f"[ERROR] Search gagal: {e}")
            return []

    # ==================================================
    # ================ UTILITY =========================
    # ==================================================
    def is_connected(self) -> bool:
        try:
            _ = self.client.list_collections()
            return True
        except Exception:
            return False


# if __name__ == "__main__":
#     db = MilvusDB("face_demo.db", dim=512)
#     emb = np.random.rand(512)
#     meta = {"name": "Alien", "timestamp": time.time()}

#     print("[TEST] Insert...")
#     record_id = db.insert(emb, meta, 0.75)
#     print("Inserted ID:", record_id)

    # # print("[TEST] Get...")
    # # data = db.get(record_id)
    # # print("Get result:", data)

    # print("[TEST] Search...")
    # start_time = time.time()
    # res = db.search(emb, top_k=3)
    # print("Search result:", res[0]["score"])
    # end_time = time.time() - start_time
    # print(f"waktu mencari di db berisi 10.000 data adalah : {end_time}")

    # count = db.client.get_collection_stats(collection_name=db.collection_name)
    # print(count)

    # # ==================================================
    # # ============ PERFORMANCE TEST: 10.000 INSERT ======
    # # ==================================================
    # print("\n[PERFORMANCE TEST] Insert 10.000 embedding unik...")

    # start_time = time.time()
    # total = 10000

    # for i in range(total):
    #     emb = np.random.rand(512)
    #     meta = {"name": f"User_{i}", "timestamp": time.time()}

    #     record_id = db.insert(emb, meta, 0.95)

    #     # Tampilkan progress setiap 1000 insert
    #     if (i + 1) % 1000 == 0:
    #         elapsed = time.time() - start_time
    #         print(f"Progress: {i+1}/{total} ({elapsed:.2f} detik)")

    # total_time = time.time() - start_time
    # print(f"\n✅ Selesai insert {total} embedding dalam {total_time:.2f} detik.")
    # print(f"Rata-rata per insert: {total_time / total:.6f} detik")

    # ==================================================
    # ============ CEK JUMLAH DATA =====================
    # ==================================================
    # stats = db.client.get_collection_stats(collection_name=db.collection_name)
    # print(f"\nTotal record di DB sekarang: {stats['row_count']}")
