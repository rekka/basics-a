import numpy as np

def jacobi(A, b):
    v = np.zeros_like(b)
    for k in range(1000):
        print('v_{:<4} = {}'.format(k, v))

        v_new = (b - A.dot(v)
                 + v * A.diagonal())\
                        / A.diagonal()

        if np.max(np.abs(v - v_new)) < 1e-10:
            break

        v = v_new
    return v


A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]])
b = np.array([6., 25., -11., 15.])

x = jacobi(A, b)
print("Solution:", x)
print("Residual:", np.dot(A, x) - b)
