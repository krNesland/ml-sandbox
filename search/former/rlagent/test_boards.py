"""
Created on 2024-11-30

Author: Kristoffer Nesland

Description: Brief description of the file
"""

from search.former.boards import load_board

test_board_0 = """
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


test_board_1 = """
0224000
0114121
0443313
3213423
3411311
2422333
3234414
4441421
3244234
"""


test_board_2 = """
0000000
0000000
0000000
0000000
0000000
0000000
3311411
3443313
1112344
"""


test_board_3 = """
0000000
0000000
0000000
0000000
0000000
0000000
0000000
0000000
4132142
"""


ALL_TEST_BOARDS_STR = [
    test_board_0,
    test_board_1,
    test_board_2,
    test_board_3,
]


ALL_TEST_BOARDS = [load_board(board_str) for board_str in ALL_TEST_BOARDS_STR]
