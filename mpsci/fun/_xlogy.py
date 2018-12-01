
import mpmath


def xlogy(x, y):
    if x == 0:
        return mpmath.mp.zero
    else:
        return x * mpmath.log(y)
