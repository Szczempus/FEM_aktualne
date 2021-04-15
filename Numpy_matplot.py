import numpy as np
import matplotlib as plt
from tabulate import tabulate


def Exc_no_1():
    arr = np.array([1, 2, 3, 4, 5])
    print(arr)
    A = np.array([[1, 2, 3], [7, 8, 9]])
    print(A)
    A = np.array([[1, 2, 3],
                  [7, 8, 9]])
    print(A)
    A = np.array([[1, 2,
                   3],
                  [7, 8, 9]])
    print(A)

    v = np.arange(1, 7)
    print(v, "\n")
    v = np.arange(-2, 7)
    print(v, "\n")
    v = np.arange(1, 10, 3)
    print(v, "\n")
    v = np.arange(1, 10.1, 3)
    print(v, "\n")
    v = np.arange(1, 11, 3)
    print(v, "\n")
    v = np.arange(1, 2, 0.1)
    print(v, "\n")

    v = np.linspace(1, 3, 4)
    print(v)
    v = np.linspace(1, 10, 4)
    print(v)

    X = np.ones((2, 3))
    Y = np.zeros((2, 3, 4))
    Z = np.eye(2)
    Q = np.random.rand(2, 5)
    print(X, "\n\n", Y, "\n\n", Z, "\n\n", Q)

    A = np.array([[1, 2, 3, 0, 0],
                  [7, 8, 9, 0, 0]])
    U = np.block([[A], [X, Z]])
    print(U)

    V = np.block([[
        np.block([
            np.block([[np.linspace(1, 3, 3)],
                      [np.zeros((2, 3))]]),
            np.ones((3, 1))])
    ],
        [np.array([100, 3, 1 / 2, 0.333])]])
    print(V)

    print(V[0, 2])
    print(V[3, 0])
    print(V[3, 3])
    print(V[-1, -1])
    print(V[-4, -3])

    print(V[3, :])
    print(V[:, 2])
    print(V[3, 0:3])
    print(V[np.ix_([0, 2, 3], [0, -1])])
    print(V[3])

    Q = np.delete(V, 2, 0)
    print(Q)
    Q = np.delete(V, 2, 1)
    print(Q)
    v = np.arange(1, 7)
    print(np.delete(v, 3, 0))

    print(np.size(v))
    print(np.shape(v))
    print(np.size(V))
    print(np.shape(V))

    A = np.array([[1, 0, 0],
                  [2, 3, -1],
                  [0, 7, 2]])
    B = np.array([[1, 2, 3],
                  [-1, 5, 2],
                  [2, 2, 2]])
    print(A + B)
    print(A - B)
    print(A + 2)
    print(2 * A)

    MM1 = A @ B
    print(MM1)
    MM2 = B @ A
    print(MM2)

    MT1 = A * B
    print(MT1)
    MT2 = B * A
    print(MT2)

    DT1 = A / B
    print(DT1)

    C = np.linalg.solve(A, MM1)
    print(C)
    x = np.ones((3, 1))
    b = A @ x
    y = np.linalg.solve(A, b)
    print(y)

    PM = np.linalg.matrix_power(A, 2)
    PT = A ** 2

    A.T
    A.transpose()
    A.conj().T
    A.conj().transpose()

    print(A == B)
    print(A != B)
    print(2 < A)
    print(A > B)
    print(A < B)
    print(A >= B)
    print(A <= B)
    np.logical_not(A)
    np.logical_and(A, B)
    np.logical_or(A, B)
    np.logical_xor(A, B)
    print(np.all(A))
    print(np.any(A))
    print(v > 4)
    print(np.logical_or(v > 4, v < 2))
    print(np.nonzero(v > 4))
    print(v[np.nonzero(v > 4)])


def Exc_no_2():
    first = np.linspace(start=1, stop=5, num=5)
    second = np.linspace(start=5, stop=1, num=5)
    third = np.zeros(shape=(3, 2))
    fourth = np.full((2, 3), 2)
    fifth = np.linspace(start=-90, stop=-70, num=3)
    sixth = np.full((5, 1), 10)

    full = np.block([[fourth], [fifth]])
    full = np.block([third, full])
    temp = np.block([[first], [second]])
    full = np.block([[temp], [full]])
    mocnyfull = np.block([full, sixth])
    print(f"{mocnyfull}\n\t")
    return mocnyfull


def Exc_no_3():
    matrix = Exc_no_2()
    matrix_add = matrix[1] + matrix[3]
    print(f"{matrix_add}\n\t")


def Exc_no_4():
    matrix = Exc_no_2()
    matrix_zip = list(map(np.max, zip(*matrix)))
    print(f"{matrix_zip}\n\t")


def choose(x):
    if x == 1:
        Exc_no_1()
    elif x == 2:
        Exc_no_2()
    elif x == 3:
        Exc_no_3()
    elif x == 4:
        Exc_no_4()
    else:
        quit()


def main():
    while True:
        print("Witaj w konsolowym nawigatorze zadań\n\t Przed tobą menu wyboru: \n\t "
              "1. Zadanie 2,\n\t "
              "2. Zadanie 3,\n\t "
              "3. Zadanie 4,\n\t "
              "4. Zadanie 5,\n\t "
              "5. Quit.")
        chose = int(input("Który skrypt wybierasz?: "))

        choose(chose)
    return


if __name__ == '__main__':
    main()
