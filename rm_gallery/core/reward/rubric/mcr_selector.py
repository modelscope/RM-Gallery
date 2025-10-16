#!/usr/bin/env python3
"""
MCR² (Maximal Coding Rate Reduction) Selector

This module implements an optimized MCR² based selection algorithm for
rubric subset selection. The algorithm maximizes coding rate to select
the most diverse and informative subset from a candidate pool.

Key Features:
- SVD-based fast coding rate computation
- Adaptive batch selection with early stopping
- Dimensionality reduction for efficiency
- Configurable selection parameters

"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from tqdm import tqdm


@dataclass
class MCR2Config:
    """
    Configuration for MCR² selection algorithm.

    Attributes:
        batch_size: Number of samples to select per iteration
        eps: Regularization parameter for coding rate computation
        normalize: Whether to normalize embeddings
        min_increment_threshold: Minimum coding rate increment to continue
        patience: Number of low increments before early stopping
        max_samples: Maximum number of samples to select
        candidate_sample_ratio: Ratio of candidates to sample for efficiency
        pca_components: Number of PCA components for dimensionality reduction
        embedding_batch_size: Batch size for embedding generation
    """

    batch_size: int = 5
    eps: float = 0.1
    normalize: bool = True
    min_increment_threshold: float = 0.001
    patience: int = 3
    max_samples: int = 100
    candidate_sample_ratio: float = 0.3
    pca_components: int = 100
    embedding_batch_size: int = 20


@dataclass
class SelectionResult:
    """Results from MCR² selection"""

    selected_indices: List[int]
    selected_texts: List[str]
    final_sample_count: int
    final_coding_rate: float
    batch_history: List[Dict[str, Any]]
    coding_rate_history: List[float]
    increment_history: List[float]
    cumulative_samples: List[int]
    analysis: Dict[str, Any]
    embeddings: np.ndarray
    configuration: Dict[str, Any]


class MCR2Selector:
    """
    MCR² based selector for optimal subset selection.

    This selector uses Maximal Coding Rate Reduction to identify the most
    diverse and informative subset from a candidate pool.

    Args:
        embedding_fn: Optional custom embedding function. If None, uses dashscope.
        embedding_dim: Dimension of embeddings (default: 1536)
        config: Default MCR2Config for selection parameters

    Example:
        >>> selector = MCR2Selector()
        >>> results = selector.select(texts, max_samples=50)
        >>> print(f"Selected {results.final_sample_count} samples")
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        embedding_dim: int = 1536,
        config: Optional[MCR2Config] = None,
    ):
        self.embedding_fn = embedding_fn
        self.embedding_dim = embedding_dim
        self.default_config = config or MCR2Config()

    def generate_embeddings(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)

        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")

        batch_size = batch_size or self.default_config.embedding_batch_size
        all_embeddings = []

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Generating embeddings",
            disable=len(texts) < batch_size,
        ):
            batch_texts = texts[i : i + batch_size]

            try:
                if self.embedding_fn:
                    # Use custom embedding function
                    embeddings = self.embedding_fn(batch_texts)
                else:
                    # Use default dashscope embedding
                    from dashscope import TextEmbedding

                    rsp = TextEmbedding.call(
                        model=TextEmbedding.Models.text_embedding_v1, input=batch_texts
                    )

                    if rsp.status_code == 200:
                        embeddings = [
                            record["embedding"] for record in rsp.output["embeddings"]
                        ]
                    else:
                        logger.warning(f"Embedding API failed: {rsp.status_code}")
                        embeddings = [np.zeros(self.embedding_dim) for _ in batch_texts]

                all_embeddings.extend(embeddings)

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                all_embeddings.extend(
                    [np.zeros(self.embedding_dim) for _ in batch_texts]
                )

        return np.array(all_embeddings)

    def compute_coding_rate(self, X: np.ndarray, eps: Optional[float] = None) -> float:
        """
        Compute coding rate using SVD decomposition.

        The coding rate R(X) measures the amount of information required to
        encode the data while preserving its structure. Higher coding rate
        indicates more diverse/informative data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            eps: Regularization parameter for numerical stability

        Returns:
            Coding rate value (in bits)
        """
        eps = eps or self.default_config.eps
        n, _ = X.shape

        if n == 0:
            return 0.0

        try:
            # Sample for efficiency if matrix is large
            if n > 50:
                sample_size = min(50, n)
                sample_idx = np.random.choice(n, size=sample_size, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X

            # SVD decomposition
            _, singular_values, _ = np.linalg.svd(X_sample, full_matrices=False)

            # Keep components that capture 95% of variance
            energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
            k = np.searchsorted(energy, 0.95) + 1
            k = min(k, len(singular_values))

            # Compute coding rate using principal singular values
            s_main = singular_values[:k]
            log_det_approx = 2 * np.sum(np.log(1 + s_main**2 / (eps**2 * n) + 1e-8))

            return float(0.5 * log_det_approx)

        except Exception as e:
            logger.warning(f"Error computing coding rate: {e}")
            return 0.0

    def select(
        self, texts: List[str], config: Optional[MCR2Config] = None, **kwargs
    ) -> SelectionResult:
        """
        Select optimal subset using MCR² algorithm.

        Args:
            texts: List of candidate texts to select from
            config: Optional MCR2Config to override defaults
            **kwargs: Additional parameters to override config
                (e.g., max_samples=50, batch_size=3)

        Returns:
            SelectionResult containing selected indices, texts, and analysis

        Raises:
            ValueError: If texts is empty

        Example:
            >>> results = selector.select(
            ...     texts=rubrics,
            ...     max_samples=100,
            ...     patience=5
            ... )
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")

        # Merge configurations
        cfg = config or self.default_config
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        logger.info(
            f"🚀 MCR² Selection: {len(texts)} candidates → {cfg.max_samples} samples"
        )

        # 1. Generate embeddings
        logger.info("Generating embeddings...")
        X = self.generate_embeddings(texts)

        # 2. Dimensionality reduction
        X = self._apply_dimensionality_reduction(X, cfg)

        # 3. Normalization
        if cfg.normalize:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # 4. Adaptive selection
        selection_data = self._adaptive_selection(X, texts, cfg)

        # 5. Create result
        result = SelectionResult(
            selected_indices=selection_data["selected_indices"],
            selected_texts=[texts[i] for i in selection_data["selected_indices"]],
            final_sample_count=len(selection_data["selected_indices"]),
            final_coding_rate=selection_data["coding_rate_history"][-1],
            batch_history=selection_data["batch_history"],
            coding_rate_history=selection_data["coding_rate_history"],
            increment_history=selection_data["increment_history"],
            cumulative_samples=selection_data["cumulative_samples"],
            analysis=self._analyze_results(
                selection_data["cumulative_samples"],
                selection_data["coding_rate_history"],
                selection_data["increment_history"],
            ),
            embeddings=X,
            configuration={
                "batch_size": cfg.batch_size,
                "eps": cfg.eps,
                "min_increment_threshold": cfg.min_increment_threshold,
                "patience": cfg.patience,
                "max_samples": cfg.max_samples,
                "candidate_sample_ratio": cfg.candidate_sample_ratio,
            },
        )

        logger.info(
            f"✅ Selection completed: {result.final_sample_count} samples, "
            f"coding_rate={result.final_coding_rate:.4f}"
        )

        return result

    def _apply_dimensionality_reduction(
        self, X: np.ndarray, cfg: MCR2Config
    ) -> np.ndarray:
        """Apply PCA dimensionality reduction if needed"""
        n_samples, original_dim = X.shape
        max_components = min(n_samples, original_dim, cfg.pca_components)

        if original_dim > max_components:
            logger.info(f"PCA: {original_dim} → {max_components} dimensions")
            pca = PCA(n_components=max_components, random_state=42)
            return pca.fit_transform(X)

        return X

    def _adaptive_selection(
        self, X: np.ndarray, texts: List[str], cfg: MCR2Config
    ) -> Dict[str, Any]:
        """Perform adaptive batch selection"""
        selected_indices = []
        candidate_indices = list(range(len(texts)))

        batch_history = []
        coding_rate_history = [0.0]
        increment_history = []
        cumulative_samples = [0]

        batch_num = 0
        low_increment_count = 0

        while len(selected_indices) < cfg.max_samples and candidate_indices:
            batch_num += 1
            current_batch_size = min(
                cfg.batch_size, cfg.max_samples - len(selected_indices)
            )

            if current_batch_size <= 0:
                break

            # Current coding rate
            R_current = (
                self.compute_coding_rate(X[selected_indices], cfg.eps)
                if selected_indices
                else 0.0
            )

            # Sample candidates for efficiency
            sampled_candidates = self._sample_candidates(
                candidate_indices, cfg.candidate_sample_ratio
            )

            # Select batch
            batch_indices = self._select_batch(
                X, selected_indices, sampled_candidates, current_batch_size, cfg.eps
            )

            if not batch_indices:
                break

            # Calculate increment
            new_selected = selected_indices + batch_indices
            R_new = self.compute_coding_rate(X[new_selected], cfg.eps)
            increment = R_new - R_current

            # Record history
            batch_history.append(
                {
                    "batch_num": batch_num,
                    "batch_indices": batch_indices,
                    "increment": increment,
                    "coding_rate": R_new,
                    "cumulative_samples": len(new_selected),
                }
            )
            coding_rate_history.append(R_new)
            increment_history.append(increment)
            cumulative_samples.append(len(new_selected))

            # Update selection
            selected_indices = new_selected
            for idx in batch_indices:
                if idx in candidate_indices:
                    candidate_indices.remove(idx)

            # Early stopping check
            if increment < cfg.min_increment_threshold:
                low_increment_count += 1
                if low_increment_count >= cfg.patience:
                    logger.info(f"Early stopping at batch {batch_num}")
                    break
            else:
                low_increment_count = 0

        return {
            "selected_indices": selected_indices,
            "batch_history": batch_history,
            "coding_rate_history": coding_rate_history,
            "increment_history": increment_history,
            "cumulative_samples": cumulative_samples,
        }

    def _sample_candidates(
        self, candidate_indices: List[int], sample_ratio: float
    ) -> List[int]:
        """Sample candidates for efficiency"""
        if len(candidate_indices) > 100:
            sample_size = max(50, int(len(candidate_indices) * sample_ratio))
            return np.random.choice(
                candidate_indices, size=sample_size, replace=False
            ).tolist()
        return candidate_indices.copy()

    def _select_batch(
        self,
        X: np.ndarray,
        selected_indices: List[int],
        candidate_indices: List[int],
        batch_size: int,
        eps: float,
    ) -> List[int]:
        """Select a batch of samples"""
        if batch_size == 1:
            return self._select_single(X, selected_indices, candidate_indices, eps)
        else:
            return self._select_diverse_batch(
                X, selected_indices, candidate_indices, batch_size
            )

    def _select_single(
        self,
        X: np.ndarray,
        selected_indices: List[int],
        candidate_indices: List[int],
        eps: float,
    ) -> List[int]:
        """Select single best sample"""
        best_delta = -np.inf
        best_idx = -1

        R_current = (
            self.compute_coding_rate(X[selected_indices], eps)
            if selected_indices
            else 0.0
        )

        # Evaluate candidates
        eval_candidates = (
            np.random.choice(
                candidate_indices, size=min(50, len(candidate_indices)), replace=False
            ).tolist()
            if len(candidate_indices) > 50
            else candidate_indices
        )

        for idx in eval_candidates:
            temp_indices = selected_indices + [idx]
            R_temp = self.compute_coding_rate(X[temp_indices], eps)
            delta = R_temp - R_current

            if delta > best_delta:
                best_delta = delta
                best_idx = idx

        return [best_idx] if best_idx != -1 else []

    def _select_diverse_batch(
        self,
        X: np.ndarray,
        selected_indices: List[int],
        candidate_indices: List[int],
        batch_size: int,
    ) -> List[int]:
        """Select diverse batch using distance heuristic"""
        batch_indices = []
        temp_candidates = candidate_indices.copy()

        # First sample: farthest from selected or max norm
        if selected_indices:
            selected_X = X[selected_indices]
            center = np.mean(selected_X, axis=0)
            distances = [
                (np.linalg.norm(X[idx] - center), idx) for idx in temp_candidates
            ]
            distances.sort(reverse=True)
            first_idx = distances[0][1]
        else:
            norms = [np.linalg.norm(X[idx]) for idx in temp_candidates]
            first_idx = temp_candidates[np.argmax(norms)]

        batch_indices.append(first_idx)
        temp_candidates.remove(first_idx)

        # Subsequent samples: farthest from batch center
        for _ in range(batch_size - 1):
            if not temp_candidates:
                break

            batch_center = np.mean(X[batch_indices], axis=0)

            # Evaluate subset of candidates
            eval_size = min(30, len(temp_candidates))
            eval_candidates = np.random.choice(
                temp_candidates, size=eval_size, replace=False
            )

            best_dist = -1
            best_idx = -1
            for idx in eval_candidates:
                dist = np.linalg.norm(X[idx] - batch_center)
                if dist > best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx != -1:
                batch_indices.append(best_idx)
                temp_candidates.remove(best_idx)

        return batch_indices

    def _analyze_results(
        self,
        cumulative_samples: List[int],
        coding_rates: List[float],
        increments: List[float],
    ) -> Dict[str, Any]:
        """Analyze selection results"""
        if len(coding_rates) < 2:
            return {
                "optimal_sample_count": cumulative_samples[-1]
                if cumulative_samples
                else 0
            }

        # Find optimal point where increment drops significantly
        optimal_point = cumulative_samples[-1]
        if increments:
            avg_increment = np.mean(increments)
            threshold = avg_increment * 0.3

            for i, inc in enumerate(increments):
                if inc < threshold:
                    optimal_point = cumulative_samples[i + 1]
                    break

        return {
            "optimal_sample_count": optimal_point,
            "total_growth": coding_rates[-1] - coding_rates[0],
            "average_increment": np.mean(increments) if increments else 0,
            "final_coding_rate": coding_rates[-1],
        }
