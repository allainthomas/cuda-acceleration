"""
To test : python3 -m pytest test_scripts/test_mul.py

With a Intel Xeon 2.20 GHz and a Tesla K80

-------------------------------------------------------------------------------------------------------- benchmark: 6 tests -------------------------------------------------------------------------------------------------------
Name (time in ms)                                    Min                    Max                   Mean              StdDev                 Median                   IQR            Outliers       OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_Matrix_Multiplication[shape0-cupy]           3.1799 (1.0)           3.3887 (1.0)           3.2303 (1.0)        0.0895 (1.0)           3.1853 (1.0)          0.0747 (1.0)           1;1  309.5676 (1.0)           5           1
test_Matrix_Multiplication[shape0-numpy]         55.5077 (17.46)        65.2538 (19.26)        59.7598 (18.50)      4.6849 (52.32)        58.0010 (18.21)        8.8271 (118.18)        1;0   16.7337 (0.05)          5           1
test_Matrix_Multiplication[shape1-cupy]         278.1504 (87.47)       290.6821 (85.78)       286.3212 (88.64)      5.1647 (57.68)       287.8290 (90.36)        7.3747 (98.74)         1;0    3.4926 (0.01)          5           1
test_Matrix_Multiplication[shape2-cupy]       2,263.3192 (711.76)    2,268.3167 (669.38)    2,265.2793 (701.26)     1.9223 (21.47)     2,265.0614 (711.10)       2.4566 (32.89)         2;0    0.4414 (0.00)          5           1
test_Matrix_Multiplication[shape1-numpy]      6,881.5478 (>1000.0)   7,022.5002 (>1000.0)   6,942.8856 (>1000.0)   60.5153 (675.79)    6,942.3977 (>1000.0)    105.5072 (>1000.0)       2;0    0.1440 (0.00)          5           1
test_Matrix_Multiplication[shape2-numpy]     54,355.3896 (>1000.0)  55,801.4011 (>1000.0)  55,051.8513 (>1000.0)  667.8756 (>1000.0)  54,874.4127 (>1000.0)  1,247.7295 (>1000.0)       2;0    0.0182 (0.00)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
=================== 6 passed, 1 warning in 391.88s (0:06:31) ===================

"""

import pytest
import importlib
import numpy
import cupy

def run_sync(m, func, *args):
    res = func(*args)
    if m.__name__ == "cupy":
        m.cuda.Device().synchronize()
    return res

def gen_data_warmup(m, compute_func, data_func, shape):
    data = run_sync(m, data_func, shape)
    run_sync(m, compute_func, data)
    return data

def run_benchmark(benchmark, m, compute_func, data_func, shape):
    data = gen_data_warmup(m, compute_func, data_func, shape)

    return benchmark.pedantic(run_sync, args=(m, compute_func, data), rounds=5)

@pytest.mark.parametrize("module", ["numpy", "cupy"])
@pytest.mark.parametrize("shape", [(1000, 1000), (5000, 5000), (10000, 10000)])
def test_Matrix_Multiplication(benchmark, module, shape):
    m = importlib.import_module(module)

    compute_func = lambda data: data.dot(data)
    run_benchmark(benchmark, m, compute_func, m.random.random, shape)