import math

import numpy as np
import platypus


class ResultWrapper:
    def __init__(self, variables, directions, index=None):
        self.problem = ProblemWrapper(variables, directions)
        self.objectives = variables
        self.index = index


class ProblemWrapper:
    def __init__(self, variables, directions):
        self.objectives = variables
        self.nconstrs = 0
        self.nobjs = len(variables)
        self.directions = directions


def get_pareto_front_idx(xs: tuple[np.ndarray, ...], directions: tuple[int, ...]) -> np.ndarray:
    # Wrap the data so that Platypus will understand it
    results = [ResultWrapper(vals, directions, index=i) for i, vals in enumerate(zip(*xs))]
    results_front = platypus.nondominated(results)
    return np.array([result.index for result in results_front])


def get_pareto_front(xs: tuple[np.ndarray, ...], directions: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    idx_front = get_pareto_front_idx(xs, directions)
    xs_front = tuple([x[idx_front] for x in xs])
    return xs_front


def get_nondominated_sort(xs: tuple[np.ndarray, ...], directions: tuple[int, ...], do_rank_shuffle: bool = True) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """
    Sort the input in a Pareto non-dominated way.
    If `do_rank_shuffle` is `True` then perform a rank-grouped shuffle.
    """
    # Wrap the data so that Platypus will understand it
    results = [ResultWrapper(vals, directions) for vals in zip(*xs)]
    platypus.nondominated_sort(results)
    ranks = np.array([result.rank for result in results])  # type: ignore
    if do_rank_shuffle:
        idx_shuffle_by_rank = arg_shuffle_per_rank(ranks)
    else:
        idx_shuffle_by_rank = np.arange(len(results))
    idx_sorted = np.argsort(ranks)
    idx_final = idx_shuffle_by_rank[idx_sorted]
    ranks_sorted = ranks[idx_final]
    xs_sorted = tuple([x[idx_final] for x in xs])
    return ranks_sorted, xs_sorted


def arg_shuffle_per_rank(ranks: np.ndarray) -> np.ndarray:
    """ Finds a vector of indices that shuffles the input per rank-group. """
    idx_rank_idx = np.arange(ranks.shape[0])
    for rank in range(ranks.max()):
        idx_rank_match = np.argwhere(ranks == rank)
        idx_rank_match_shuffled = idx_rank_match.copy()
        np.random.shuffle(idx_rank_match_shuffled)
        idx_rank_idx[idx_rank_match] = idx_rank_idx[idx_rank_match_shuffled]
    return idx_rank_idx


def construct_min_pareto(x: np.ndarray) -> np.ndarray:
    """
    Constructs a vector that can be used for Pareto front plotting. This vector
    contains the guaranteed Pareto front.
    """
    x_mat = np.concatenate([x[:-1, np.newaxis], x[1:, np.newaxis]], axis=1)
    x_min = x_mat.min(axis=1)
    x_mat2 = np.concatenate([x[:-1, np.newaxis], x_min[:, np.newaxis]], axis=1)
    x_out: np.ndarray = np.concatenate([x_mat2.reshape(-1), x[[-1]]])
    return x_out


def calculate_objective(*xs: np.ndarray) -> np.ndarray:
    return math.prod(xs)  # type: ignore


def find_best_solution(*xs: np.ndarray) -> int:
    objective = calculate_objective(*xs)
    return int(np.argmax(objective))


if __name__ == "__main__":
    n = 1000
    x = np.random.randn(n)
    y = np.random.randn(n)

    directions = (platypus.Problem.MAXIMIZE, platypus.Problem.MAXIMIZE)

    ranks_sort, (x_sort, y_sort) = get_nondominated_sort((x, y), directions)

    n_top = 50

    import matplotlib.pyplot as plt

    plt.figure()

    plt.scatter(x, y, label="all")
    plt.scatter(x_sort[:n_top], y_sort[:n_top], label=f"top {n_top}")

    plt.legend()
    plt.grid()

    plt.show()
