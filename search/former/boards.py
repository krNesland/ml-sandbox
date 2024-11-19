import numpy as np


b_111124 = """
1233321
3243224
2141313
4333243
1342142
2424423
4324411
4324224
2433331
"""


b_121124 = """
1224124
1114121
1443313
3213423
3411311
2422333
3234414
4441421
3244234
"""


b_131124 = """
2113232
3331314
2414131
1121224
3143341
1323431
3311411
3443313
1112344
"""


b_191124 = """
3341324
1221334
2323221
3421132
1434142
2323113
3333321
1232441
4132142
"""


def load_board(board_str: str) -> np.ndarray:
    return np.array([[int(cell) for cell in row.strip()] for row in board_str.strip().split("\n")])


if __name__ == "__main__":
    board = load_board(b_111124)
    print(board)
