# ==============================================================================
# faiss_manager.py — FAISS Vector Index Management
# ==============================================================================
# Features:
#   - Creation of Inner Product (Cosine Similarity) index
#   - Disk persistence (save/load)
#   - Metadata storage alongside vectors (FAISS only stores vectors)
#   - Corruption detection via dimension checks
#   - Automatic index rebuilding if corruption is detected
# ==============================================================================

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple

from src.config import FAISS_INDEX_DIR
from src.logger import get_logger

logger = get_logger("FAISS")


class FAISSManager:
    """
    Manages the FAISS vector index and associated metadata.

    Attributes:
        index (faiss.Index): The FAISS index instance.
        metadata (List[Dict]): List of metadata dicts corresponding to vectors.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize the FAISS manager.

        Args:
            dimension: dimensionality of the embeddings (384 for MiniLM-L6).
        """
        self.dimension = dimension
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._index_path = FAISS_INDEX_DIR / "faiss.index"
        self._metadata_path = FAISS_INDEX_DIR / "metadata.pkl"

    def create_index(self):
        """Create a fresh IndexFlatIP (Inner Product) for cosine similarity."""
        logger.info("Creating new FAISS index (dim=%d)", self.dimension)
        # Inner Product is equivalent to Cosine Similarity when vectors are normalized
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def load_index(self) -> bool:
        """
        Load index and metadata from disk.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not self._index_path.exists() or not self._metadata_path.exists():
            logger.info("No existing index found on disk.")
            self.create_index()
            return False

        try:
            logger.info("Loading FAISS index from %s", self._index_path)
            self.index = faiss.read_index(str(self._index_path))

            logger.info("Loading metadata from %s", self._metadata_path)
            with open(self._metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

            # ── Corruption Check ─────────────────────────────────────────────
            if self.index.d != self.dimension:
                logger.error(
                    "Index dimension mismatch: expected %d, got %d. Rebuilding.",
                    self.dimension, self.index.d
                )
                self.create_index()
                return False

            if self.index.ntotal != len(self.metadata):
                logger.error(
                    "Index count mismatch: vectors=%d, metadata=%d. Corrupt. Rebuilding.",
                    self.index.ntotal, len(self.metadata)
                )
                self.create_index()
                return False

            logger.info("Index loaded successfully: %d vectors.", self.index.ntotal)
            return True

        except Exception as e:
            logger.error("Failed to load index: %s. Creating new one.", e)
            self.create_index()
            return False

    def save_index(self):
        """Persist index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save.")
            return

        try:
            faiss.write_index(self.index, str(self._index_path))
            with open(self._metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info("Index saved (vectors=%d).", self.index.ntotal)
        except Exception as e:
            logger.error("Failed to save index: %s", e)

    def add_vectors(self, embeddings: np.ndarray, meta_list: List[Dict[str, Any]]):
        """
        Add vectors and their metadata to the index.

        Args:
            embeddings: Numpy array of shape (N, dimension).
            meta_list: List of N metadata dictionaries.
        """
        if self.index is None:
            self.create_index()

        if len(embeddings) != len(meta_list):
            raise ValueError(
                f"Count mismatch: embeddings={len(embeddings)}, metadata={len(meta_list)}"
            )

        if len(embeddings) == 0:
            return

        # Ensure correct type/shape
        embeddings = embeddings.astype(np.float32)

        try:
            self.index.add(embeddings)
            self.metadata.extend(meta_list)
            logger.info("Added %d vectors to index. Total: %d",
                        len(embeddings), self.index.ntotal)
        except Exception as e:
            logger.error("Failed to add vectors to FAISS: %s", e)
            raise

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        """
        Search the index for the top-k nearest neighbors.

        Args:
            query_vector: Numpy array of shape (1, dimension).
            k: Number of results to return.

        Returns:
            distances: Numpy array of similarity scores.
            results_meta: List of metadata dicts for the retrieved vectors.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty. Returning 0 results.")
            return np.array([]), []

        try:
            # D = distances (scores), I = indices
            D, I = self.index.search(query_vector.astype(np.float32), k)

            results_meta = []
            valid_scores = []
            
            # Retrieve metadata for the returned indices
            # Flatten the result arrays (since query is a single vector)
            indices = I[0]
            scores = D[0]

            for idx, score in zip(indices, scores):
                if idx == -1:  # FAISS returns -1 if fewer than k results exist
                    continue
                if idx < len(self.metadata):
                    results_meta.append(self.metadata[idx])
                    valid_scores.append(score)

            return np.array(valid_scores), results_meta

        except Exception as e:
            logger.error("FAISS search failed: %s", e)
            return np.array([]), []
