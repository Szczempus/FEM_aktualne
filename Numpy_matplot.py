import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import zadanie_15


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

    x = np.arange(0.0, 2.0, 0.01)
    y1 = np.sin(2.0 * np.pi * x)
    y2 = np.cos(2.0 * np.pi * x)
    plt.plot(x, y1, 'r:', x, y2, 'g')
    plt.legend(('dane y1', 'dane y2'))
    plt.xlabel('Czas')
    plt.ylabel('Pozycja')
    plt.title('Wykres ')
    plt.grid(True)
    plt.show()

    x = np.arange(0.0, 2.0, 0.01)
    y1 = np.sin(2.0 * np.pi * x)
    y2 = np.cos(2.0 * np.pi * x)
    y = y1 * y2
    l1, = plt.plot(x, y, 'b')
    l2, l3 = plt.plot(x, y1, 'r:', x, y2, 'g')
    plt.legend((l2, l3, l1), ('dane y1', 'dane y2', 'y1 * y2'))
    plt.xlabel('Czas')
    plt.ylabel('Pozycja')
    plt.title('Wykres')
    plt.grid(True)
    plt.show()


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
    B = matrix[1] + matrix[3]
    print(f"{B}\n\t")
    return B


def Exc_no_5():
    matrix = Exc_no_2()
    C = list(map(np.max, zip(*matrix)))
    print(C)
    return C


def Exc_no_6():
    matrix = Exc_no_3()
    D = np.delete(matrix, 0)
    D = np.delete(D, len(D) - 1)
    print(D)
    return D


def Exc_no_7():
    matrix = Exc_no_6()
    matrix[matrix == 4] = 0
    print(matrix)
    return matrix


def Exc_no_8():
    matrix = Exc_no_5()
    minimum = min(matrix)
    maximum = max(matrix)
    E = [val for val in matrix if val != minimum and val != maximum]
    print(E)
    return E


def Exc_no_9():
    matrix = Exc_no_2()
    minimum = matrix.min()
    maximum = matrix.max()
    for val in range(np.shape(matrix)[0]):
        if minimum in matrix[val, :]:
            if maximum in matrix[val, :]:
                print(matrix[val, :])


def Exc_no_10():
    matrix_1 = Exc_no_6()
    matrix_2 = Exc_no_8()
    multiply_1 = matrix_1 * matrix_2
    multiply_2 = matrix_1 @ matrix_2
    print(multiply_1)
    print(multiply_2)
    return multiply_1, multiply_2


def Exc_no_11():
    try:
        size = int(input("Podaj rozmiar macierzy: "))
        matrix = np.random.randint(0, 11, [size, size])
        print(matrix)
        print(np.trace(matrix))
        return matrix, np.trace(matrix)
    except:
        print("Wartość nie jest poprawna, wprowadź ponownie")
        return


def Exc_no_12():
    try:
        size = int(input("Podaj rozmiar macierzy: "))
        matrix = np.random.randint(0, 11, [size, size])
        next_matrix = matrix * (1 - np.eye(size, size))
        next_matrix = next_matrix * (1 - np.fliplr(np.eye(size, size)))
        print(next_matrix)
        return next_matrix
    except:
        print("Wartość nie jest poprawna, wprowadź ponownie")
        return


def Exc_no_13():
    try:
        size = int(input("Podaj rozmiar macierzy: "))
    except:
        print("Wartość nie jest poprawna, wprowadź ponownie")
        return
    matrix = np.random.randint(0, 11, [size, size])
    suma = 0
    for i in range(size):
        if i % 2 == 0:
            suma = suma + np.sum(matrix[i, :])
    print(suma)
    return suma


def Exc_no_14(x):
    funkcja = lambda x: np.cos(2 * x)
    y = funkcja(x)
    plt.plot(x, y, 'r--')
    plt.show()
    return y


def Exc_no_15(x):
    y1 = Exc_no_14(x)
    y2 = zadanie_15.Exc_no_15(x)
    plt.plot(x, y2, 'g+', x, y1, 'r--')
    plt.show()
    return y2


def Exc_no_17(x):
    y3 = 3 * Exc_no_14(x) + zadanie_15.Exc_no_15(x)
    plt.plot(x, y3, 'b*')
    plt.show()
    return y3


def Exc_no_18():
    matrix = np.array([[10, 5, 1, 7], [10, 9, 5, 5], [1, 6, 7, 3], [10, 0, 1, 5]])
    score = np.array([[34], [44], [25], [27]])
    X = np.linalg.inv(matrix) @ score
    print(X)


def choose(x):
    if x == 2:
        Exc_no_1()
    elif x == 3:
        Exc_no_2()
    elif x == 4:
        Exc_no_3()
    elif x == 5:
        Exc_no_5()
    elif x == 6:
        Exc_no_6()
    elif x == 7:
        Exc_no_7()
    elif x == 8:
        Exc_no_8()
    elif x == 9:
        Exc_no_9()
    elif x == 10:
        Exc_no_10()
    elif x == 11:
        Exc_no_11()
    elif x == 12:
        Exc_no_12()
    elif x == 13:
        Exc_no_13()
    elif x == 14:
        pocz, kon, krok = input("Wprowadź po przecnku początek, koniec przedziału i krok: ").split(',')
        pocz = float(pocz)
        kon = float(kon)
        krok = float(krok)
        x = np.arange(pocz, kon, krok)
        Exc_no_14(x)
    elif x == 15:
        pocz, kon, krok = input("Wprowadź po przecnku początek, koniec przedziału i krok: ").split(',')
        pocz = float(pocz)
        kon = float(kon)
        krok = float(krok)
        x = np.arange(pocz, kon, krok)
        Exc_no_15(x)
    elif x == 17:
        pocz, kon, krok = input("Wprowadź po przecnku początek, koniec przedziału i krok: ").split(',')
        pocz = float(pocz)
        kon = float(kon)
        krok = float(krok)
        x = np.arange(pocz, kon, krok)
        Exc_no_17(x)
    elif x == 18:
        Exc_no_18()
    else:
        quit()


def main():
    while True:
        print("Witaj w konsolowym nawigatorze zadań\n\t Przed tobą menu wyboru: \n\t "
              "2. Exc_no_2,\n\t "
              "3. Exc_no_3,\n\t "
              "4. Zadanie 4,\n\t "
              "5. Zadanie 5,\n\t "
              "5. Quit.")
        chose = int(input("Który skrypt wybierasz?: "))

        choose(chose)
    return


if __name__ == '__main__':
    main()
