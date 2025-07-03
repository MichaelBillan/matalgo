import numpy as np
import time

EPSILON = 1e-10

def swap_rows(matrix, row1, row2):
    if row1 != row2:
        matrix[[row1, row2]] = matrix[[row2, row1]]

def my_abs(x):
    return x if x >= 0 else -x

def pivot_index(mat, col, start):
    col_slice = mat[start:, col]
    max_idx = 0
    max_val = my_abs(col_slice[0])
    for i in range(1, len(col_slice)):
        v = my_abs(col_slice[i])
        if v > max_val:
            max_val = v
            max_idx = i
    return start + max_idx


def row_has_nonzero(row):
    for v in row:
        if my_abs(v) > EPSILON:
            return True
    return False

def rank(mat):

    m, n = mat.shape
    row  = 0

    start = time.perf_counter()

    for j in range(n):
        if row >= m:
            break
    
        pivot_row_index = pivot_index(mat, j, row)
        if my_abs(mat[pivot_row_index, j]) < EPSILON:
            continue

        swap_rows(mat, row, pivot_row_index)
        mat[row, :] /= mat[row, j]

        if row + 1 < m:
            factors = mat[row+1:, j:j+1]
            mat[row+1:, :] -= factors * mat[row:row+1, :]

        row += 1

    elapsed = time.perf_counter() - start
    print(f"rank() finished in {elapsed:.6f} s")

    return row


def det(A):

    U = A.copy().astype(np.float64)
    n = U.shape[0]

    sign = 1.0
    prod_diag = 1.0

    start = time.perf_counter()

    for k in range(n):

        abs_col = []
        for i in range(k, n):
            abs_col.append(my_abs(U[i, k]))

        pivot_off = 0
        pivot_val = abs_col[0]

        for i in range(1, len(abs_col)):
            v = abs_col[i]
            if v > pivot_val:
                pivot_off = i
                pivot_val = v
        p = k + pivot_off

        if abs(U[p, k]) < EPSILON:
            return 0.0

        if p != k:
            U[[k, p]] = U[[p, k]]
            sign = -sign

        pivot = U[k, k]
        prod_diag *= pivot

        if k + 1 == n:
            break

        f = U[k + 1:, k] / pivot
        U[k + 1:, k + 1:] -= f[:, None] * U[k, k + 1:]
        U[k + 1:, k] = 0.0

    elapsed = time.perf_counter() - start
    print(f"det() finished in {elapsed:.6f} s")

    return sign * prod_diag



def inverse(mat):
    n = mat.shape[0]
    aug = np.hstack((mat.copy().astype(np.float64), np.eye(n)))

    start = time.perf_counter()

    for k in range(n):
        p = pivot_index(aug, k, k)

        if my_abs(aug[p, k]) < EPSILON:
            raise ValueError("Matrix is singular")
        
        swap_rows(aug, k, p)
        aug[k, :] /= aug[k, k]
        factors = aug[k + 1:, k:k + 1]
        aug[k + 1:, :] -= factors * aug[k:k + 1, :]

    for k in range(n - 1, -1, -1):
        factors = aug[:k, k:k + 1]
        aug[:k, :] -= factors * aug[k:k + 1, :]

    elapsed = time.perf_counter() - start
    print(f"inverse() finished in {elapsed:.6f} s")

    return aug[:, n:]


A = np.loadtxt("rank_data.txt")
B = np.loadtxt("det_data.txt")
C = np.loadtxt("inv_data.txt")

rank = rank(A)
determinant = det(B)
inversion = inverse(C)

print("matrix rank =", rank)
print("determinant =", determinant)   
