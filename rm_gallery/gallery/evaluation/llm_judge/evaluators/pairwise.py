"""
Pairwise Evaluator for VERL Integration

Supports two modes:
1. winrate: Simple win rate calculation
2. dgr: DGR algorithm (= TFAS, Tournament Feedback Arc Set)
"""

import concurrent.futures
import itertools
from collections import defaultdict
from typing import Callable, Dict, List, Literal, Tuple

from rm_gallery.core.model.base import BaseLLM
from rm_gallery.gallery.evaluation.llm_judge.templates.base import VERLPromptTemplate


class PairwiseEvaluator:
    """
    Pairwise evaluator supporting four modes:

    1. winrate: Simple win rate (wins / total_comparisons)
    2. copeland: Copeland method (net wins = wins - losses)
    3. dgr: DGR algorithm (= TFAS, resolves cycles)
    4. elo: ELO Rating algorithm
    """

    def __init__(
        self,
        llm: BaseLLM,
        template: VERLPromptTemplate,
        mode: Literal["winrate", "copeland", "dgr", "elo"] = "dgr",
        max_workers: int = 10,
        verbose: bool = False,
        elo_initial_rating: float = 1500.0,
        elo_k_factor: float = 32.0,
        elo_max_iterations: int = 100,
        elo_convergence_threshold: float = 0.01,
        llm_timeout: float = 30.0,  # 每次LLM调用的超时时间（秒）
        max_retries: int = 2,  # 失败时的最大重试次数
        **kwargs,
    ):
        self.llm = llm
        self.template = template
        self.mode = mode
        self.max_workers = max_workers
        self.verbose = verbose
        self.elo_initial_rating = elo_initial_rating
        self.elo_k_factor = elo_k_factor
        self.elo_max_iterations = elo_max_iterations
        self.elo_convergence_threshold = elo_convergence_threshold
        self.llm_timeout = llm_timeout  # 超时控制
        self.max_retries = max_retries  # 重试控制
        self.kwargs = kwargs

    def evaluate(
        self, prompt: str, responses: List[str], reference: str = None, **kwargs
    ) -> Dict:
        """
        Execute pairwise evaluation

        Process:
        1. Pairwise compare all response pairs (using LLM)
        2. Calculate final scores based on mode:
           - winrate: simple win rate
           - dgr: TFAS algorithm (resolve conflicts)
        """
        n = len(responses)

        if n == 0:
            return {"scores": [], "comparisons": []}

        if n == 1:
            return {"scores": [0.0], "comparisons": []}

        # 1. Pairwise compare all pairs (concurrent LLM calls)
        comparisons = self._pairwise_compare_all(prompt, responses, reference)

        # 2. Calculate scores based on mode
        if self.mode == "winrate":
            scores, extra_info = self._calculate_winrate_scores(comparisons, n)
        elif self.mode == "copeland":
            scores, extra_info = self._calculate_copeland_scores(comparisons, n)
        elif self.mode == "dgr":
            scores, extra_info = self._calculate_dgr_scores(comparisons, n)
        elif self.mode == "elo":
            scores, extra_info = self._calculate_elo_scores(comparisons, n)
        else:
            raise ValueError(f"Unknown pairwise mode: {self.mode}")

        return {
            "scores": scores,
            "comparisons": comparisons,
            "n_responses": n,
            "pairwise_mode": self.mode,
            **extra_info,
        }

    def _pairwise_compare_all(
        self, prompt: str, responses: List[str], reference: str
    ) -> List[Dict]:
        """
        Pairwise compare all response pairs (using RM-Gallery LLM)

        优化版：添加超时控制和非阻塞处理

        Returns:
            [
                {'i': 0, 'j': 1, 'result': 'A', 'winner': 0},
                {'i': 0, 'j': 2, 'result': 'B', 'winner': 2},
                ...
            ]
        """
        comparisons = []
        n = len(responses)

        # Generate all pairs to compare
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        if self.verbose:
            print(
                f"[Pairwise] Comparing {len(pairs)} pairs with timeout={self.llm_timeout}s, max_workers={self.max_workers}"
            )

        # Concurrent LLM calls with timeout control
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_pair = {
                executor.submit(
                    self._compare_pair_with_retry,
                    prompt,
                    responses[i],
                    responses[j],
                    reference,
                    i,
                    j,
                ): (i, j)
                for i, j in pairs
            }

            # 使用超时控制避免无限等待
            total_timeout = (
                self.llm_timeout * (len(pairs) // self.max_workers + 1) + 10.0
            )

            try:
                for future in concurrent.futures.as_completed(
                    future_to_pair, timeout=total_timeout
                ):
                    i, j = future_to_pair[future]
                    try:
                        # 为每个future设置超时
                        comparison = future.result(timeout=self.llm_timeout + 5.0)
                        comparisons.append(comparison)
                    except concurrent.futures.TimeoutError:
                        if self.verbose:
                            print(
                                f"⚠️  Timeout comparing {i} vs {j} after {self.llm_timeout}s"
                            )
                        # Default to tie on timeout
                        comparisons.append(
                            {
                                "i": i,
                                "j": j,
                                "result": "tie",
                                "winner": None,
                                "error": "timeout",
                            }
                        )
                    except Exception as e:
                        if self.verbose:
                            print(f"❌ Error comparing {i} vs {j}: {e}")
                        # Default to tie on error
                        comparisons.append(
                            {
                                "i": i,
                                "j": j,
                                "result": "tie",
                                "winner": None,
                                "error": str(e),
                            }
                        )
            except concurrent.futures.TimeoutError:
                if self.verbose:
                    print(f"⚠️  Total comparison timeout after {total_timeout}s")
                # 取消剩余的任务
                for future in future_to_pair:
                    future.cancel()
                # 对未完成的pairs添加tie结果
                completed_pairs = {(comp["i"], comp["j"]) for comp in comparisons}
                for i, j in pairs:
                    if (i, j) not in completed_pairs:
                        comparisons.append(
                            {
                                "i": i,
                                "j": j,
                                "result": "tie",
                                "winner": None,
                                "error": "total_timeout",
                            }
                        )

        return comparisons

    def _compare_pair_with_retry(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        reference: str,
        idx_a: int,
        idx_b: int,
    ) -> Dict:
        """
        使用LLM比较一对响应（带重试机制）
        """
        import time

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._compare_pair(
                    prompt, response_a, response_b, reference, idx_a, idx_b
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    if self.verbose:
                        print(
                            f"⚠️  Retry {attempt + 1}/{self.max_retries} for pair ({idx_a}, {idx_b}): {e}"
                        )
                    time.sleep(0.5 * (attempt + 1))  # 指数退避
                    continue
                else:
                    if self.verbose:
                        print(f"❌ All retries failed for pair ({idx_a}, {idx_b}): {e}")
                    # 所有重试都失败，返回tie
                    return {
                        "i": idx_a,
                        "j": idx_b,
                        "result": "tie",
                        "winner": None,
                        "error": f"max_retries_exceeded: {str(last_error)}",
                    }

    def _compare_pair(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        reference: str,
        idx_a: int,
        idx_b: int,
    ) -> Dict:
        """
        Use LLM to compare one pair of responses
        """
        # Generate prompt
        comparison_prompt = self.template.generate_prompt(
            user_query=prompt,
            response_a=response_a,
            response_b=response_b,
            reference=reference or "",
        )

        # Call LLM (using RM-Gallery's unified interface)
        from rm_gallery.core.model.message import ChatMessage, MessageRole

        messages = [ChatMessage(role=MessageRole.USER, content=comparison_prompt)]

        try:
            response = self.llm.chat(messages=messages)
            llm_output = response.message.content
        except Exception as e:
            if self.verbose:
                print(f"LLM call failed: {e}")
            llm_output = "error"

        # Parse result
        result = self.template.parse_result(llm_output)

        # Determine winner
        if result == "a":
            winner = idx_a
        elif result == "b":
            winner = idx_b
        else:  # tie or invalid
            winner = None

        return {
            "i": idx_a,
            "j": idx_b,
            "result": result,
            "winner": winner,
            "llm_output": llm_output,
        }

    def _calculate_winrate_scores(
        self, comparisons: List[Dict], n: int
    ) -> Tuple[List[float], Dict]:
        """
        Simple win rate calculation: wins / total_comparisons

        Returns normalized scores in [-1, 1]
        """
        win_counts = [0] * n
        total_counts = [0] * n

        for comp in comparisons:
            i, j = comp["i"], comp["j"]
            winner = comp["winner"]

            total_counts[i] += 1
            total_counts[j] += 1

            if winner == i:
                win_counts[i] += 1
            elif winner == j:
                win_counts[j] += 1
            # tie: no win count increase

        # Calculate win rates
        winrates = []
        for i in range(n):
            if total_counts[i] > 0:
                winrate = win_counts[i] / total_counts[i]
            else:
                winrate = 0.5  # default medium
            winrates.append(winrate)

        # Normalize to [-1, 1]
        min_wr = min(winrates)
        max_wr = max(winrates)

        if max_wr - min_wr < 1e-6:
            scores = [0.0] * n
        else:
            scores = [2 * (wr - min_wr) / (max_wr - min_wr) - 1 for wr in winrates]

        extra_info = {
            "win_counts": win_counts,
            "total_counts": total_counts,
            "winrates": winrates,
        }

        return scores, extra_info

    def _calculate_copeland_scores(
        self, comparisons: List[Dict], n: int
    ) -> Tuple[List[float], Dict]:
        """
        Copeland method: net wins (wins - losses)

        Simple and fast method based on net win count.
        Does not resolve cycles, just counts wins minus losses.

        Returns normalized scores in [-1, 1]
        """
        win_counts = [0] * n
        loss_counts = [0] * n
        tie_counts = [0] * n

        for comp in comparisons:
            i, j = comp["i"], comp["j"]
            winner = comp["winner"]

            if winner == i:
                win_counts[i] += 1
                loss_counts[j] += 1
            elif winner == j:
                win_counts[j] += 1
                loss_counts[i] += 1
            else:  # tie
                tie_counts[i] += 1
                tie_counts[j] += 1

        # Calculate net wins
        net_wins = [win_counts[i] - loss_counts[i] for i in range(n)]

        # Normalize to [-1, 1]
        max_abs_net_wins = max(abs(nw) for nw in net_wins) if net_wins else 0

        if max_abs_net_wins < 1e-6:
            scores = [0.0] * n
        else:
            scores = [nw / max_abs_net_wins for nw in net_wins]

        extra_info = {
            "win_counts": win_counts,
            "loss_counts": loss_counts,
            "tie_counts": tie_counts,
            "net_wins": net_wins,
        }

        return scores, extra_info

    def _calculate_dgr_scores(
        self, comparisons: List[Dict], n: int
    ) -> Tuple[List[float], Dict]:
        """
        DGR algorithm = TFAS (Tournament Feedback Arc Set)

        Core idea:
        1. Build tournament graph (directed graph)
        2. Detect cycles (conflicts)
        3. Remove minimum edge set to eliminate cycles (TFAS)
        4. Calculate net wins based on conflict-free graph
        5. Normalize to [-1, 1]

        This is the core DGR algorithm (migrated from dgr_core.py TFAS implementation)
        """
        # Build comparator function from comparisons
        comparison_matrix = self._build_comparison_matrix(comparisons, n)

        def comparator(i: int, j: int) -> int:
            """Comparator function: returns 1 (i>j), -1 (i<j), 0 (tie)"""
            return comparison_matrix[i][j]

        # Build tournament graph
        graph = self._build_tournament_graph(n, comparator)

        # TFAS algorithm: remove minimum edge set to eliminate cycles
        graph, removed_edges = self._tfas_remove_conflicts(graph, n)

        # Calculate net wins based on conflict-free graph
        net_wins = self._calculate_net_wins(graph, n)

        # Normalize to [-1, 1]
        max_abs = max(abs(nw) for nw in net_wins) if net_wins else 0

        if max_abs < 1e-6:
            scores = [0.0] * n
        else:
            scores = [nw / max_abs for nw in net_wins]
            scores = [max(-1.0, min(1.0, s)) for s in scores]

        extra_info = {
            "net_wins": net_wins,
            "removed_edges_count": len(removed_edges),
            "graph_edges_count": sum(len(neighbors) for neighbors in graph.values()),
        }

        return scores, extra_info

    def _build_comparison_matrix(
        self, comparisons: List[Dict], n: int
    ) -> List[List[int]]:
        """
        Build comparison matrix from pairwise comparisons

        matrix[i][j] = 1 if i > j, -1 if i < j, 0 if tie
        """
        matrix = [[0] * n for _ in range(n)]

        for comp in comparisons:
            i, j = comp["i"], comp["j"]
            winner = comp["winner"]

            if winner == i:
                matrix[i][j] = 1
                matrix[j][i] = -1
            elif winner == j:
                matrix[i][j] = -1
                matrix[j][i] = 1
            else:  # tie
                matrix[i][j] = 0
                matrix[j][i] = 0

        return matrix

    def _build_tournament_graph(
        self, n: int, comparator: Callable[[int, int], int]
    ) -> Dict:
        """
        Build tournament graph (directed graph)

        Returns:
            graph: {node: {neighbor: {'weight': 1, 'is_tie': False}}}
        """
        graph = defaultdict(dict)

        if self.verbose:
            print(f"\n🔧 Building tournament graph (n={n})")

        for i in range(n):
            for j in range(i + 1, n):
                result = comparator(i, j)

                if result > 0:  # i > j
                    graph[i][j] = {"weight": 1, "is_tie": False, "flipped": False}
                elif result < 0:  # i < j
                    graph[j][i] = {"weight": 1, "is_tie": False, "flipped": False}
                else:  # result == 0, tie
                    # For tie, randomly choose direction but mark as tie
                    import random

                    if random.random() < 0.5:
                        graph[i][j] = {"weight": 0, "is_tie": True, "flipped": False}
                    else:
                        graph[j][i] = {"weight": 0, "is_tie": True, "flipped": False}

        if self.verbose:
            edge_count = sum(len(adj) for adj in graph.values())
            print(f"✓ Tournament graph built: {edge_count} edges")

        return graph

    def _tfas_remove_conflicts(self, graph: Dict, n: int) -> Tuple[Dict, List]:
        """
        TFAS algorithm: remove minimum edge set to eliminate all cycles

        This is the core DGR algorithm (Tournament Feedback Arc Set)

        Returns:
            (cleaned_graph, removed_edges)
        """
        if self.verbose:
            print(f"\n🎯 Starting TFAS conflict removal (n={n})")

        if n == 1:
            return defaultdict(dict), []

        # Choose algorithm based on scale
        if n <= 10:
            # Exact algorithm for small scale
            if self.verbose:
                print("Using exact TFAS algorithm")
            removed_edges = self._tfas_exact(graph, n)
        else:
            # Greedy algorithm for large scale
            if self.verbose:
                print("Using greedy TFAS algorithm")
            removed_edges = self._tfas_greedy(graph, n)

        # Apply edge removals
        final_graph = self._apply_edge_removals(graph, removed_edges)

        if self.verbose:
            print(f"✓ TFAS completed: removed {len(removed_edges)} edges")

        return final_graph, removed_edges

    def _tfas_exact(self, graph: Dict, n: int) -> List:
        """Exact TFAS algorithm for small scale (n <= 10)"""
        # Enumerate all possible topological orderings
        all_permutations = list(itertools.permutations(range(n)))

        if self.verbose:
            print(f"Evaluating {len(all_permutations)} topological orderings")

        best_removed_edges = []
        best_removal_cost = float("inf")

        # Find ordering with minimum removal cost
        for permutation in all_permutations:
            removal_cost, removed_edges = self._calculate_removal_cost(
                graph, permutation, n
            )

            if removal_cost < best_removal_cost:
                best_removal_cost = removal_cost
                best_removed_edges = removed_edges

        if self.verbose:
            print(f"Best removal cost: {best_removal_cost:.2f}")

        return best_removed_edges

    def _tfas_greedy(self, graph: Dict, n: int) -> List:
        """Greedy TFAS algorithm for large scale (n > 10)"""
        # Calculate net wins for each node
        net_wins = [0.0] * n
        out_weights = [0.0] * n
        in_weights = [0.0] * n

        for u in range(n):
            for v in graph[u]:
                edge_info = graph[u][v]
                weight = edge_info["weight"]

                out_weights[u] += weight
                in_weights[v] += weight

        # Calculate net wins and sort
        for i in range(n):
            net_wins[i] = out_weights[i] - in_weights[i]

        indexed_net_wins = [(net_wins[i], i) for i in range(n)]
        indexed_net_wins.sort(reverse=True)
        greedy_permutation = tuple([item[1] for item in indexed_net_wins])

        # Calculate removal cost
        removal_cost, removed_edges = self._calculate_removal_cost(
            graph, greedy_permutation, n
        )

        if self.verbose:
            print(f"Greedy removal cost: {removal_cost:.2f}")

        return removed_edges

    def _calculate_removal_cost(
        self, graph: Dict, permutation: Tuple, n: int
    ) -> Tuple[float, List]:
        """
        Calculate removal cost for a given topological ordering

        Returns:
            (total_removal_cost, removed_edges)
        """
        # Create position mapping: earlier position = higher priority
        position = {node: i for i, node in enumerate(permutation)}

        total_removal_cost = 0.0
        removed_edges = []

        # Check each edge against topological ordering
        for u in range(n):
            for v in graph[u]:
                edge_info = graph[u][v]

                # If u → v exists but u comes after v in ordering, need to remove
                if position[u] > position[v]:  # u has lower priority but has edge to v
                    # Calculate removal cost
                    if edge_info["is_tie"]:
                        removal_cost = 0.1  # Very low cost for tie edges
                    else:
                        removal_cost = edge_info[
                            "weight"
                        ]  # Cost = weight for non-tie edges

                    total_removal_cost += removal_cost
                    removed_edges.append((u, v, edge_info))

        return total_removal_cost, removed_edges

    def _apply_edge_removals(self, graph: Dict, removed_edges: List) -> Dict:
        """
        Apply edge removals to generate final acyclic graph

        Returns:
            Graph after removals
        """
        # Deep copy graph
        final_graph = defaultdict(dict)

        # First copy all original edges
        for u in graph:
            for v in graph[u]:
                final_graph[u][v] = graph[u][v].copy()

        # Apply removals: directly delete conflict edges
        for u, v, edge_info in removed_edges:
            # Delete edge u → v
            if u in final_graph and v in final_graph[u]:
                del final_graph[u][v]

        return final_graph

    def _calculate_net_wins(self, graph: Dict, n: int) -> List[float]:
        """
        Calculate net wins for each node based on directed graph (out-degree - in-degree)
        """
        net_wins = [0.0] * n

        for u in graph:
            for v in graph[u]:
                weight = graph[u][v]["weight"]
                net_wins[u] += weight
                net_wins[v] -= weight

        return net_wins

    def _calculate_elo_scores(
        self, comparisons: List[Dict], n: int
    ) -> Tuple[List[float], Dict]:
        """
        ELO Rating algorithm for pairwise comparisons

        Process:
        1. Initialize all ratings to initial_rating
        2. For each comparison, calculate expected win probability
        3. Update ratings based on actual result
        4. Iterate until convergence
        5. Normalize to [-1, 1]

        Returns:
            (normalized_scores, extra_info)
        """
        if self.verbose:
            print(f"\n🎯 Starting ELO rating calculation (n={n})")

        # Initialize ratings
        ratings = [self.elo_initial_rating] * n

        # Collect all comparisons with results
        comparison_results = []
        for comp in comparisons:
            i, j = comp["i"], comp["j"]
            winner = comp["winner"]

            if winner == i:
                score_i, score_j = 1.0, 0.0
            elif winner == j:
                score_i, score_j = 0.0, 1.0
            else:  # tie
                score_i, score_j = 0.5, 0.5

            comparison_results.append((i, j, score_i, score_j))

        # Iteratively update ratings
        for iteration in range(self.elo_max_iterations):
            old_ratings = ratings.copy()

            # Process all comparisons
            for i, j, score_i, score_j in comparison_results:
                # Calculate expected scores using ELO formula
                expected_i = 1.0 / (1.0 + 10 ** ((ratings[j] - ratings[i]) / 400.0))
                expected_j = 1.0 / (1.0 + 10 ** ((ratings[i] - ratings[j]) / 400.0))

                # Update ratings
                ratings[i] += self.elo_k_factor * (score_i - expected_i)
                ratings[j] += self.elo_k_factor * (score_j - expected_j)

            # Check convergence
            max_change = max(abs(ratings[i] - old_ratings[i]) for i in range(n))

            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: max_change = {max_change:.4f}")

            if max_change < self.elo_convergence_threshold:
                if self.verbose:
                    print(f"✓ Converged at iteration {iteration}")
                break

        if self.verbose:
            print(f"Final ELO ratings: {[f'{r:.1f}' for r in ratings]}")

        # Normalize to [-1, 1]
        min_rating = min(ratings)
        max_rating = max(ratings)

        if max_rating - min_rating < 1e-6:
            scores = [0.0] * n
        else:
            scores = [
                2 * (r - min_rating) / (max_rating - min_rating) - 1 for r in ratings
            ]

        extra_info = {
            "elo_ratings": ratings,
            "elo_iterations": iteration + 1,
            "elo_converged": max_change < self.elo_convergence_threshold,
        }

        return scores, extra_info
