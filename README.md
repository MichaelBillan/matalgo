# Matrix Algorithms from Scratch (Python, Only Basic NumPy)

This project implements core linear algebra algorithms for **matrix rank**, **determinant**, and **inverse** using only basic NumPy array features (no advanced NumPy matrix algorithms or shortcuts). All steps of the algorithms are written manually to closely mimic the textbook algorithms and avoid any “black box” calls.

## Features

* **Rank Calculation:**
  Computes the rank of a matrix via manual Gaussian elimination and row reduction with partial pivoting.
  No use of built-in rank functions or advanced operations.

* **Determinant Calculation:**
  Calculates the determinant of a square matrix via manual LU-like decomposition with full pivot selection and tracking for sign changes (row swaps). Returns `0.0` for singular matrices.

* **Matrix Inversion:**
  Computes the inverse of a square matrix using the Gauss-Jordan elimination method, with explicit row swaps for stability. Raises a `ValueError` for singular (non-invertible) matrices.

* **All core steps are implemented from scratch:**
  Only `np.hstack`, `np.eye`, basic slicing, and elementary arithmetic/vectorization are used for efficiency—no high-level matrix algorithms from NumPy or SciPy.

* **Performance Benchmarks:**
  Each function prints its execution time using Python's `time.perf_counter()` for performance comparison.

## File Structure

* `main.py`:
  Contains all algorithm implementations, utility functions, and a basic example of loading matrices from text files and running the algorithms.

* *Matrix data files:*
  Not included in this repo for privacy; expects `rank_data.txt`, `det_data.txt`, `inv_data.txt` to be in the working directory.

## Usage Example

```python
import numpy as np
from main import rank, det, inverse

A = np.loadtxt("rank_data.txt")
B = np.loadtxt("det_data.txt")
C = np.loadtxt("inv_data.txt")

print("matrix rank =", rank(A.copy()))
print("determinant =", det(B.copy()))
print("inverse matrix:\n", inverse(C.copy()))
```

*The `.copy()` calls ensure the input files remain unchanged, as the algorithms modify the matrix in place.*

## Restrictions and Motivation

* **No black-box matrix functions:**
  All algorithms are implemented using only basic NumPy operations, avoiding any built-in matrix rank, determinant, or inverse functions.

* **Educational Purpose:**
  The project is ideal for learning how fundamental matrix algorithms work under the hood and for benchmarking pure Python vectorized code on large datasets.

* **Numerical Stability:**
  Partial pivoting is used for stability, but beware of edge cases with nearly singular matrices (as with any manual elimination implementation).

## How to Run

1. Prepare your data files (`rank_data.txt`, `det_data.txt`, `inv_data.txt`) with numeric matrices (plain text, whitespace-separated).
2. Update the paths in `main.py` or your script as needed (recommend using relative paths).
3. Run the script:

   ```
   python main.py
   ```

   Output will include the computed rank, determinant, and inverse, as well as timing information for each algorithm.

## Notes

* This code is for learning and demonstration; for production work, prefer using NumPy's or SciPy’s optimized linear algebra functions.
* Matrix data files must be well-formed (for inversion/determinant: square and non-singular).

---

## Credits

Written by Michael Billan, implementing core matrix algorithms from first principles using vectorized NumPy only.

---

