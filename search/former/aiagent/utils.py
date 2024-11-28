from copy import deepcopy

import search.former.aiagent.scoring as scoring
from search.former.clusters import get_unique_clusters
from search.former.former import Former


def suggest_cluster(
    former: Former, scorer: scoring.ScoreBase, depth: int, width: int
) -> tuple[int, float]:
    if (depth == 0) or former.is_grid_empty():
        cluster_masks = get_unique_clusters(former.grid)
        return -1, scorer(former.grid, cluster_masks)

    cluster_masks = get_unique_clusters(former.grid)

    cluster_id_to_immediate_score: list[tuple[int, float]] = []

    # Using the immediate score to narrow down the search space in a greedy manner
    for cluster_id, cluster_mask in enumerate(cluster_masks):
        former_copy = deepcopy(former)
        former_copy.remove_shapes(cluster_mask)
        former_copy.apply_gravity()  # Not always needed. Will depend on the score function
        cluster_masks_post = get_unique_clusters(former_copy.grid)
        cluster_id_to_immediate_score.append(
            (cluster_id, scorer(former_copy.grid, cluster_masks_post))
        )

    # Sort the clusters by score in descending order. Adding some noise to be less greedy about it
    cluster_id_to_immediate_score = sorted(
        [(cluster_id, score) for cluster_id, score in cluster_id_to_immediate_score],
        key=lambda x: x[1],
        reverse=True,
    )

    cluster_id_to_score: list[tuple[int, float]] = []

    for cluster_id, _ in cluster_id_to_immediate_score[:width]:
        former_copy = deepcopy(former)
        former_copy.remove_shapes(cluster_masks[cluster_id])
        former_copy.apply_gravity()
        cluster_id_to_score.append(
            (cluster_id, suggest_cluster(former_copy, scorer, depth - 1, width)[1])
        )

    cluster_id_to_score = sorted(
        cluster_id_to_score,
        key=lambda x: x[1],
        reverse=True,
    )

    return cluster_id_to_score[0]
