def is_integer(value):
    if not isinstance(value, int):
        return False

    return True


def all_integers(array):
    for value in array:
        if not isinstance(value, int):
            return False

    return True


def is_number(value):
    if not isinstance(value, (int, float)):
        return False

    return True


def all_numbers(array):
    for value in array:
        if not isinstance(value, (int, float)):
            return False

    return True
