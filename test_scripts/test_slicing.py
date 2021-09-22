"""
To test : python3 -m pytest test_scripts/test_slicing.py

With a Intel Xeon 2.20 GHz and a Tesla K80

------------------------------------------------------------------------------------------------------- benchmark: 6 tests -------------------------------------------------------------------------------------------------------
Name (time in us)                            Min                     Max                   Mean                 StdDev                 Median                    IQR            Outliers         OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_Array_Slicing[shape0-cupy]          84.8570 (1.0)          356.9970 (1.0)         160.4378 (1.0)         113.8862 (1.0)         100.4850 (1.0)         116.3605 (1.0)           1;0  6,232.9451 (1.0)           5           1
test_Array_Slicing[shape0-numpy]        365.8680 (4.31)       1,408.6870 (3.95)        596.2656 (3.72)        454.7574 (3.99)        389.3190 (3.87)        293.1782 (2.52)          1;1  1,677.1050 (0.27)          5           1
test_Array_Slicing[shape1-cupy]       1,098.0840 (12.94)      1,517.4140 (4.25)      1,201.9870 (7.49)        177.9676 (1.56)      1,136.9690 (11.31)       145.0895 (1.25)          1;1    831.9558 (0.13)          5           1
test_Array_Slicing[shape2-cupy]       4,083.7700 (48.13)      4,752.4090 (13.31)     4,228.3036 (26.35)       293.1901 (2.57)      4,105.1050 (40.85)       183.0238 (1.57)          1;1    236.5015 (0.04)          5           1
test_Array_Slicing[shape1-numpy]     15,886.1090 (187.21)    25,787.9040 (72.24)    18,329.6284 (114.25)    4,183.8600 (36.74)    16,686.7060 (166.06)    2,682.8890 (23.06)         1;1     54.5565 (0.01)          5           1
test_Array_Slicing[shape2-numpy]     65,192.7850 (768.27)   100,983.0990 (282.87)   74,767.2466 (466.02)   14,885.1557 (130.70)   68,465.3110 (681.35)   12,939.3335 (111.20)        1;1     13.3748 (0.00)          5           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
========================= 6 passed, 1 warning in 2.96s =========================

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
def test_Array_Slicing(benchmark, module, shape):
    m = importlib.import_module(module)

    compute_func = lambda data: data[::3].copy()
    run_benchmark(benchmark, m, compute_func, m.random.random, shape)