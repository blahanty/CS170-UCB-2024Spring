import os
import pickle
import random

import networkx as nx


def improved_tsp_approximation(matrix):
    """
        An algorithm for solving the Metric TSP using minimum spanning trees, depth first search and local
        search.

        Args:
            matrix: List[List[float]]
                An n x n matrix of distances, where M[i, j] corresponds to the distance from city i to city j.

        Returns:
            path: List[int]
                A list corresponding to the order in which to visit cities, starting from path[0] and ending
                at path[-1] before returning to path[0].
    """
    n = len(matrix)

    def tsp_greedy(matrix, home):
        """
        A greedy implementation of TSP, starting and ending at home.

        Args:
            matrix: List[List[float]]
                An n x n matrix of distances, where M[i, j] corresponds to the distance from city i to city j.
            home: int
                The index of the city to start and end at.

        Returns:
            path: List[int]
                A list corresponding to the order in which to visit cities, starting from path[0] and ending
                at path[-1] before returning to path[0]. path[0] should be home.
        """
        path = [home]
        visited = [False] * n
        visited[home] = True
        current = home
        while len(path) < n:
            best = None
            best_dist = float('inf')
            for i in range(n):
                if not visited[i] and matrix[current][i] < best_dist:
                    best = i
                    best_dist = matrix[current][i]

            path.append(best)
            visited[best] = True
            current = best
        return path

    def tsp_greedy_general(matrix):
        """
        A generalized greedy implementation of TSP.

        Args:
            matrix: List[List[float]]
                An n x n matrix of distances, where M[i, j] corresponds to the distance from city i to city j.

        Returns:
            path: List[int]
                A list corresponding to the order in which to visit cities, starting from path[0] and ending
                at path[-1] before returning to path[0].
        """
        min_path = None
        min_cost = float('inf')
        for i in range(n):
            path = tsp_greedy(matrix, i)
            cost = validate_tour(path, matrix)
            if 0 < cost < min_cost:
                min_path = path
                min_cost = cost

        return min_path, min_cost

    def metric_tsp_approximation(matrix):
        """
            An algorithm for solving the Metric TSP using minimum spanning trees and depth first search.

            Args:
                matrix: List[List[float]]
                    An n x n matrix of distances, where M[i, j] corresponds to the distance from city i to city j.

            Returns:
                path: List[int]
                    A list corresponding to the order in which to visit cities, starting from path[0] and ending
                    at path[-1] before returning to path[0].
        """
        G = nx.Graph()
        for i in range(n):
            for j in range(n):
                G.add_edge(i, j, weight=matrix[i][j])

        T = nx.minimum_spanning_tree(G)
        min_path = None
        min_cost = float('inf')
        for i in range(n):
            path_pre = list(nx.dfs_preorder_nodes(T, i))
            cost_pre = validate_tour(path_pre, matrix)
            path_post = list(nx.dfs_postorder_nodes(T, i))
            cost_post = validate_tour(path_post, matrix)
            if cost_pre == -1 and cost_post == -1:
                continue
            elif cost_post == -1 or cost_pre < cost_post:
                path = path_pre
                cost = cost_pre
            else:
                path = path_post
                cost = cost_post

            if 0 < cost < min_cost:
                min_path = path
                min_cost = cost

        return min_path, min_cost

    def local_search(matrix, path):
        """
        A local search algorithm for improving the path.

        Args:
            matrix: List[List[float]]
                An n x n matrix of distances, where M[i, j] corresponds to the distance from city i to city j.
            path: List[int]
                A list corresponding to the order in which to visit cities, starting from path[0] and ending
                at path[-1] before returning to path[0].

        Returns:
            path: List[int]
                A list corresponding to the order in which to visit cities, starting from path[0] and ending
                at path[-1] before returning to path[0].
        """
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    left_prev = path[i - 1]
                    left = path[i]
                    right = path[j - 1]
                    right_next = path[j]
                    new_path_1 = path[:i] + path[i:j][::-1] + path[j:]
                    cost_diff_1 = matrix[left_prev][right] + matrix[left][right_next]
                    cost_diff_1 -= matrix[left_prev][left] + matrix[right][right_next]
                    new_path_2 = path[:i][::-1] + path[i:j] + path[j:][::-1]
                    if i > 1:
                        cost_diff_2 = matrix[path[0]][left] + matrix[right][path[-1]] + matrix[right_next][left_prev]
                        cost_diff_2 -= matrix[left_prev][left] + matrix[right][right_next] + matrix[path[0]][path[-1]]
                    else:
                        cost_diff_2 = 1

                    if cost_diff_1 < cost_diff_2:
                        new_path = new_path_1
                        cost_diff = cost_diff_1
                    else:
                        new_path = new_path_2
                        cost_diff = cost_diff_2

                    if cost_diff < 0:
                        path = new_path
                        improved = True
                        break

                if improved:
                    break

        return path

    def random_search(matrix, path, iterations):
        for _ in range(iterations):
            i, j, k = sorted(random.sample(range(1, n - 1), 3))
            left_prev = path[i - 1]
            left = path[i]
            mid_prev = path[j - 1]
            mid = path[j]
            right_prev = path[k - 1]
            right = path[k]
            curr_cost = matrix[left_prev][left] + matrix[mid_prev][mid] + matrix[right_prev][right]
            new_cost_1 = matrix[left_prev][mid] + matrix[right_prev][left] + matrix[mid_prev][right]
            new_cost_2 = matrix[left_prev][right_prev] + matrix[mid][left] + matrix[mid_prev][right]
            new_cost_3 = matrix[left_prev][mid] + matrix[right_prev][mid_prev] + matrix[left][right]
            new_cost_4 = matrix[left_prev][mid_prev] + matrix[left][right_prev] + matrix[mid][right]
            new_cost_5 = matrix[left_prev][right] + matrix[path[-1]][mid] + matrix[right_prev][left] + matrix[mid_prev][
                path[0]] - matrix[path[0]][path[-1]]
            min_cost = min(new_cost_1, new_cost_2, new_cost_3, new_cost_4, new_cost_5)
            if min_cost < curr_cost:
                if min_cost == new_cost_1:
                    path = path[:i] + path[j:k] + path[i:j] + path[k:]
                elif min_cost == new_cost_2:
                    path = path[:i] + path[j:k][::-1] + path[i:j] + path[k:]
                elif min_cost == new_cost_3:
                    path = path[:i] + path[j:k] + path[i:j][::-1] + path[k:]
                elif min_cost == new_cost_4:
                    path = path[:i] + path[i:j][::-1] + path[j:k][::-1] + path[k:]
                else:
                    path = path[:i] + path[k:] + path[j:k] + path[i:j]

        return path

    tour_greedy, cost_greedy = tsp_greedy_general(matrix)
    tour_approx, cost_approx = metric_tsp_approximation(matrix)
    tour = tour_greedy if cost_greedy < cost_approx else tour_approx
    tour = local_search(matrix, tour)
    cost = validate_tour(tour, matrix)
    return random_search(matrix, tour, int(1.5 * cost))


def validate_tour(tour, matrix):
    """
    Provided function to verify the validity of your TSP approximation function.
    Returns the length of the tour if it is valid, -1 otherwise.
    Feel free to use or modify this function however you please,
    as the autograder will only call your tsp_approximation function.
    """
    n = len(tour)
    cost = 0
    for i in range(n):
        if matrix[tour[i - 1]][tour[i]] == float("inf"):
            return -1
        cost += matrix[tour[i - 1]][tour[i]]
    return cost


def verify_basic(matrix, path):
    """Verify that the proposed solution is valid."""
    assert len(path) == len(
        matrix
    ), f"There are {len(matrix)} cities but your path has {len(path)} cities!"
    assert sorted(path) == list(
        range(len(path))
    ), f"Your path is not a permutation of cities (ints from 0 to {len(path) - 1})"


def evaluate_tsp(tsp_approximation):
    """
    Provided function to evaluate your TSP approximation function.
    Feel free to use or modify this function however you please,
    as the autograder will only call your tsp_approximation function.
    """

    test_cases = pickle.load(open(os.path.join("tsp_cases.pkl"), "rb"))

    total_cost = 0
    for i, case in enumerate(test_cases["files"] + test_cases["generated"]):
        tour = tsp_approximation(case)
        verify_basic(case, tour)
        cost = validate_tour(tour, case)
        assert cost != -1
        total_cost += cost
        print(f"Case {i}: {cost}")

    print(f"Total cost: {total_cost}")
    return total_cost


if __name__ == "__main__":
    evaluate_tsp(improved_tsp_approximation)
