
from timeit import timeit

class PadeApproximations:

    @staticmethod
    def e(x):
        return (1 + x*(0.5 + x*(0.1111 + x*(0.01389 + x*(0.00099 + 0.000033 * x ))))) / \
            (1 + x * (-0.5 + x * (0.1111 + x * (-0.01389 + x * (0.00099 - 0.000033 * x)))))

    @staticmethod
    def e_derivative(x):
        return (x*(x*(x*(x*(x*(x*(x*(x*(x*1.77636*10**-15 + 60.) - 1.98952*10**-13) - 5054.55) - 3.27418*10**-11) + 409879.) - 8.14907*10**-10) - 2.54913*10**7) + 7.45058*10**-9) + 9.18274*10**8)/\
            (x*(x*(x*(x*(x - 30.) + 420.909) - 3366.67) + 15151.5) - 30303.)**2

    @staticmethod
    def e_fast(x):
        x1 = 0.5*x
        return (1. + x1) / (1. - x1)

    @staticmethod
    def e_fast_derivative(x):
        return 4. / (x-2.)**2

    @staticmethod
    def sigmoid(x):
        return 1./(1.+PadeApproximations.e(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sig = PadeApproximations.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def sigmoid_fast(x):
        return 1. /(1.+PadeApproximations.e_fast((-x)))

    @staticmethod
    def sigmoid_fast_derivative(x):
        sig = PadeApproximations.sigmoid_fast(x)
        return sig * (1 - sig)





def test_pade():
    print(PadeApproximations.e(1))
    print(PadeApproximations.e_fast(1))
    print(timeit("PadeApproximations.e(1)", number=10_000, globals=globals()))
    print(timeit("PadeApproximations.e_fast(1)", number=10_000, globals=globals()))

    print("Derivative".center(50,'-'))
    print(PadeApproximations.e_derivative(1))
    print(PadeApproximations.e_fast_derivative(1))
    print(timeit("PadeApproximations.e_derivative(1)", number=10_000, globals=globals()))
    print(timeit("PadeApproximations.e_fast_derivative(1)", number=10_000, globals=globals()))

    print("Sigmoid".center(50,'-'))
    print(PadeApproximations.sigmoid(1))
    print(PadeApproximations.sigmoid_fast(1))
    print(timeit("PadeApproximations.sigmoid(1)", number=10_000, globals=globals()))
    print(timeit("PadeApproximations.sigmoid(1)", number=10_000, globals=globals()))

    print("Sigmoid Derivative".center(50,'-'))
    print(PadeApproximations.sigmoid_derivative(1))
    print(PadeApproximations.sigmoid_fast_derivative(1))
    print(timeit("PadeApproximations.sigmoid_derivative(1)", number=10_000, globals=globals()))
    print(timeit("PadeApproximations.sigmoid_fast_derivative(1)", number=10_000, globals=globals()))
