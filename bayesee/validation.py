def all_integers(tuple):
    for value in tuple:
        if not isinstance(value, int):
            return False
    
    return True
