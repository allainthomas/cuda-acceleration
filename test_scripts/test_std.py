"""
To test : python3 -m pytest test_scripts/test_std.py

With a Intel Xeon 2.20 GHz and a Tesla K80

------------------------------------------------------------------------------------------------- benchmark: 6 tests -------------------------------------------------------------------------------------------------
Name (time in ms)                                Min                   Max                  Mean            StdDev                Median               IQR            Outliers       OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_Standard_Deviation[shape0-numpy]         2.0836 (1.0)          2.3518 (1.0)          2.1853 (1.0)      0.1082 (3.48)         2.1482 (1.0)      0.1555 (6.04)          1;0  457.6099 (1.0)           5           1
test_Standard_Deviation[shape0-cupy]         41.3142 (19.83)       44.6455 (18.98)       43.3289 (19.83)    1.7489 (56.29)       44.5831 (20.75)    3.1359 (121.81)        2;0   23.0793 (0.05)          5           1
test_Standard_Deviation[shape1-numpy]       119.9924 (57.59)      124.9272 (53.12)      122.0779 (55.86)    2.4052 (77.41)      120.9545 (56.30)    4.4762 (173.87)        1;0    8.1915 (0.02)          5           1
test_Standard_Deviation[shape2-numpy]       470.9057 (226.01)     482.9942 (205.37)     478.0845 (218.78)   4.6560 (149.84)     479.8170 (223.35)   6.0084 (233.39)        2;0    2.0917 (0.00)          5           1
test_Standard_Deviation[shape1-cupy]        793.1399 (380.66)     793.2112 (337.27)     793.1790 (362.97)   0.0311 (1.0)        793.1782 (369.22)   0.0556 (2.16)          2;0    1.2607 (0.00)          5           1
test_Standard_Deviation[shape2-cupy]      3,171.9368 (>1000.0)  3,172.0338 (>1000.0)  3,171.9696 (>1000.0)  0.0372 (1.20)     3,171.9596 (>1000.0)  0.0257 (1.0)           1;1    0.3153 (0.00)          5           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
======================== 6 passed, 1 warning in 31.14s =========================

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
def test_Standard_Deviation(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.std, m.random.random, shape)