from copy import deepcopy
import random

from search.former.game import Former
import search.former.scoring as scoring

def suggest_cluster(former: Former, depth: int, width: int) -> tuple[int, int]:
    if (depth == 0) or former.is_grid_empty():
        return -1, scoring.score_grid_by_num_removed(former.grid)

    clusters = former.get_unique_clusters()

    cluster_id_to_immediate_score: list[tuple[int, int]] = []

    # Using the immediate score to narrow down the search space in a greedy manner
    for cluster_id, cluster in enumerate(clusters):
        former_copy = deepcopy(former)
        former_copy.remove_shapes(x=cluster[0][0], y=cluster[0][1])
        former_copy.apply_gravity()  # Not always needed. Will depend on the score function
        cluster_id_to_immediate_score.append((cluster_id, scoring.score_grid_by_num_removed(former_copy.grid)))

    # Sort the clusters by score in descending order. Adding some noise to be less greedy about it
    cluster_id_to_immediate_score = sorted(
        [(cluster_id, score + random.randint(0, 4)) for cluster_id, score in cluster_id_to_immediate_score],
        key=lambda x: x[1],
        reverse=True,
    )

    cluster_id_to_score: list[tuple[int, int]] = []

    for cluster_id, _ in cluster_id_to_immediate_score[:width]:
        former_copy = deepcopy(former)
        former_copy.remove_shapes(x=clusters[cluster_id][0][0], y=clusters[cluster_id][0][1])
        former_copy.apply_gravity()
        cluster_id_to_score.append((cluster_id, suggest_cluster(former_copy, depth - 1, width)[1]))

    cluster_id_to_score = sorted(cluster_id_to_score, key=lambda x: x[1], reverse=True,)
        
    return cluster_id_to_score[0]
