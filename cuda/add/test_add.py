import ctypes
from ctypes import c_float, POINTER
import numpy as np
import time
import random

lib = ctypes.CDLL("./cuda/add/add.dll") 
rf = ctypes.byref
"""
Build command (rtx 4090)
nvcc --shared add.cu -o add.dll -Xcompiler "/MD" -arch=sm_86
"""

lib.add_cuda.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float)]
lib.add_cuda.restype = None

lib.multiply_cuda.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float)]
lib.multiply_cuda.restype = None

lib.mat_add_cuda.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int
]

lib.mat_add_cuda.restype = None

rows, cols = 5120, 5120

A = np.random.rand(rows, cols).astype(np.float32)
B = np.random.rand(rows, cols).astype(np.float32)
C = np.zeros((rows, cols), dtype=np.float32)

a = c_float(3.5)
b = c_float(2.25)
result = c_float()

lib.add_cuda(rf(a), rf(b), ctypes.byref(result))
print("Result from add CUDA:", result.value)

lib.multiply_cuda(rf(a), rf(b), ctypes.byref(result))
print("Result from multiply CUDA:", result.value)

s = time.time()
lib.mat_add_cuda(
    A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    rows,
    cols
)
end = time.time()
print(end-s)

mat1 = []
mat2 = []
mat3 = []
for y in range(rows):
    mat1.append([])
    mat2.append([])
    mat3.append([])
    for x in range(cols):
        mat1[-1].append(random.randrange(0,4))
        mat2[-1].append(random.randrange(0,4))
s2 = time.time()
for y in range(rows):
    for x in range(cols):   
        mat3.append(mat1[y][x]+mat2[y][x])
print(time.time()-s2)

print("A (first 5x5):\n", A[:5, :5])
print("B (first 5x5):\n", B[:5, :5])
print("C (first 5x5):\n", C[:5, :5])