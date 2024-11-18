from search.former.game import Former
import search.former.scoring as scoring
import search.former.bot as bot
import search.former.boards as boards
import time
from multiprocessing import Pool

from typing import Tuple, List


# Function to suggest a cluster
def suggest(former: Former) -> Tuple[int, float]:
    cluster_id, score = bot.suggest_cluster(former, depth=5, width=7)
    return cluster_id, score

# Main game function
def play_game(former: Former):
    clusters = former.get_unique_clusters()
    print("Initial Grid:")
    former.print_grid(clusters=clusters)
    print(f"Grid score: {scoring.score_grid_by_num_removed(former.grid)}")
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

            # Use Pool to parallelize the suggestion process
            with Pool(processes=8) as pool:  # Adjust the number of processes based on your CPU cores
                results: List[Tuple[int, float]] = pool.map(suggest, [former] * 8)

            for cluster_id, score in results:
                print(f"Cluster: {cluster_id} | Score: {score}")

            response = input("Enter the row and column to click (e.g., 3 4), or q to quit: ")
            if response == "q":
                break

            x, y = map(int, response.split())
            if not former.is_in_bounds(x, y):
                print("🚫 Invalid coordinates. Try again.")
                continue
            if former.grid[x][y] == 0:
                print("🚫 Selected cell is empty. Try again.")
                continue

            # Remove shapes and apply gravity
            former.remove_shapes(x, y)
            former.apply_gravity()

            # Display updated grid
            clusters = former.get_unique_clusters()
            print("Updated Grid:")
            former.print_grid(clusters=clusters)
            print(f"⭐️ Grid score: {scoring.score_grid_by_num_removed(former.grid)}")
            former.plot_grid(clusters=clusters)

            turn_num += 1
        
        except ValueError:
            print("🚫 Invalid input. Please enter two integers separated by a space.")


if __name__ == "__main__":
    # ROWS = 9
    # COLS = 7
    # SHAPES = [1, 2, 3, 4]  # Use numbers to represent different shapes
    # former = Former(rows=ROWS, cols=COLS, shapes=SHAPES)

    board = boards.load_board(boards.b_131124)

    former = Former.from_board(board)

    play_game(former)