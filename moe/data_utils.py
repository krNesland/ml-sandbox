"""
Created 04 February 2024
Kristoffer Nesland, kristoffernesland@gmail.com
"""


def get_train_data():
    raise NotImplementedError()


def get_eval_data():
    raise NotImplementedError()


def main():
    train_x_tensor, train_y_tensor = get_train_data()

    print(train_x_tensor)
    print(train_y_tensor)


if __name__ == "__main__":
    main()
