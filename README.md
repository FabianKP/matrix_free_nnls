 matrix_free_nnls
 ===
 
This repository contains code to solve non-negative least-squares problems of the form

$$
\min_{x \geq 0} ||Ax - b||_2^2
$$

using the accelerated projected gradient descent method with restart described 
[here](https://angms.science/doc/NMF/nnls_pgd.pdf).

The matrix $A$ can be provided by the user as `numpy.ndarray` or as `scipy.sparse.linalg.LinearOperator`.
The latter is recommended for large-scale problems.

![image](https://github.com/FabianKP/matrix_free_nnls/blob/main/examples/comparison.png)

Usage
---

```python
import numpy as np
from matrix_free_nnls import solve_nnls

a = np.random.randn(100, 100)
b = np.random.randn(100)

x, res = solve_nnls(a, b)
```

Installation
---

You can download the wheel file 
[matrix_free_nnls-1.0-py3-none-any.whl](https://github.com/FabianKP/matrix_free_nnls/blob/main/dist/matrix_free_nnls-1.0-py3-none-any.whl) and then pip-install:
```bash
pip install matrix_free_nnls-1.0-py3-none-any.whl
```