__UUID: int = -1


def get():
    global __UUID
    __UUID += 1
    return __UUID
