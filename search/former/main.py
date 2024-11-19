from search.former.game import Former
import search.former.scoring as scoring
import search.former.bot as bot
import search.former.boards as boards
import time
from multiprocessing import Pool
from search.former.clusters import get_unique_clusters, is_in_bounds

from typing import Tuple, List


# Function to suggest a cluster
def suggest(args: Tuple[Former, scoring.ScoreBase]) -> Tuple[int, float]:
    former, scorer = args
    cluster_id, score = bot.suggest_cluster(former, scorer, depth=4, width=7)
    return cluster_id, score

# Main game function
def play_game(former: Former, scorer: scoring.ScoreBase) -> None:
    clusters = get_unique_clusters(former.grid)
    print("Initial Grid:")
    former.print_grid(clusters=clusters)
    print(f"Grid score: {scorer(former.grid):.2f}")
    former.plot_grid(clusters=clusters)

    turn_num = 1

    while True:
        print("\n\n")
        if former.is_grid_empty():
            print("🔥 Congratulations! All shapes have been removed.")
            break

        try:
            print(f"🥊 Turn {turn_num}")

            print("Getting move suggestions...")

            # Commented out now for simplicity
            # # Use Pool to parallelize the suggestion process
            # with Pool(processes=8) as pool:  # Adjust the number of processes based on your CPU cores
            #     results: List[Tuple[int, float]] = pool.map(suggest, [(former, scorer)] * 8)

            start_time = time.time()
            results = [suggest((former, scorer)) for _ in range(1)]
            end_time = time.time()

            print(f"Time taken: {end_time - start_time:.4f} seconds")

            for cluster_id, score in results:
                print(f"Cluster: {cluster_id} | Score: {score:.2f}")

            response = input("Enter the row and column to click (e.g., 3 4), or q to quit: ")
            if response == "q":
                break

            x, y = map(int, response.split())
            if not is_in_bounds(x, y, rows=former.rows, cols=former.cols):
                print("🚫 Invalid coordinates. Try again.")
                continue
            if former.grid[x, y] == 0:
                print("🚫 Selected cell is empty. Try again.")
                continue

            # Remove shapes and apply gravity
            former.remove_shapes(x, y)
            former.apply_gravity()

            # Display updated grid
            clusters = get_unique_clusters(former.grid)
            print("Updated Grid:")
            former.print_grid(clusters=clusters)
            print(f"⭐️ Grid score: {scorer(former.grid):.2f}")
            former.plot_grid(clusters=clusters)

            turn_num += 1
        
        except ValueError:
            print("🚫 Invalid input. Please enter two integers separated by a space.")


if __name__ == "__main__":
    # ROWS = 9
    # COLS = 7
    # SHAPES = [1, 2, 3, 4]  # Use numbers to represent different shapes
    # former = Former(rows=ROWS, cols=COLS, shapes=SHAPES)

    board = boards.load_board(boards.b_191124)
    # scorer = scoring.ScoreGridByNumRemoved()
    scorer = scoring.ScoreGridByNumClusters()

    former = Former.from_board(board)

    play_game(former, scorer)