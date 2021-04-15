import numpy as np
import matplotlib as plt


def gradient(x, y, z):
    return x, y, z


def Henholtz(u, x, y, z, c):
    f = gradient(x, y, z) ** 2 * u + c * u
    return f


def main():
    return


if __name__ == '__main__':
    main()
