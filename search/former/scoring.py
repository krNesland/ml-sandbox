def score_grid_by_num_removed(grid: list[list[int]]) -> int:
    return sum([cell == 0 for row in grid for cell in row])