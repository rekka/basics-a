import numpy as np

def gauss_seidel(A, b):
    v = np.zeros_like(b)
    for k in range(1000):
        print('v_{:<4} = {}'.format(k, v))

        v_old = np.copy(v)
        for i in range(A.shape[0]):
            s1 = A[i, :i].dot(v[:i])
            s2 = A[i, i + 1:].dot(v[i + 1:])
            v[i] = (b[i] - s1 - s2) / A[i, i]

        if np.max(np.abs(v - v_old)) < 1e-10:
            break
    return v

A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]])
b = np.array([6., 25., -11., 15.])

v = gauss_seidel(A, b)
print("Solution:", v)
print("Residual:", np.dot(A, v) - b)
