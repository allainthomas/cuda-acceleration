"""
To test : python3 -m pytest test_scripts/test_elmt.py

With a Intel Xeon 2.20 GHz and a Tesla K80

----------------------------------------------------------------------------------------------- benchmark: 6 tests ----------------------------------------------------------------------------------------------
Name (time in ms)                         Min                   Max                  Mean             StdDev                Median                IQR            Outliers       OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_Elementwise[shape0-cupy]          1.1204 (1.0)          1.2089 (1.0)          1.1578 (1.0)       0.0355 (1.0)          1.1541 (1.0)       0.0552 (1.0)           2;0  863.6708 (1.0)           5           1
test_Elementwise[shape1-cupy]         23.5526 (21.02)       25.9575 (21.47)       25.0401 (21.63)     1.2209 (34.39)       25.9050 (22.45)     2.1444 (38.85)         1;0   39.9360 (0.05)          5           1
test_Elementwise[shape0-numpy]        34.3125 (30.63)       38.7799 (32.08)       36.4807 (31.51)     1.9684 (55.44)       36.4253 (31.56)     3.6054 (65.33)         2;0   27.4117 (0.03)          5           1
test_Elementwise[shape2-cupy]         81.8690 (73.07)       87.0630 (72.02)       84.4247 (72.92)     2.4547 (69.14)       83.1141 (72.01)     4.3245 (78.36)         3;0   11.8449 (0.01)          5           1
test_Elementwise[shape1-numpy]       921.5405 (822.51)   1,004.8488 (831.22)     951.1883 (821.51)   34.1758 (962.63)     938.2263 (812.93)   48.5757 (880.13)        1;0    1.0513 (0.00)          5           1
test_Elementwise[shape2-numpy]     3,722.2555 (>1000.0)  3,857.4439 (>1000.0)  3,769.6900 (>1000.0)  52.5950 (>1000.0)  3,754.3202 (>1000.0)  59.0465 (>1000.0)       1;0    0.2653 (0.00)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
======================== 6 passed, 1 warning in 32.64s =========================

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
def test_Elementwise(benchmark, module, shape):
    m = importlib.import_module(module)

    compute_func = lambda data: m.sin(data) ** 2 + m.cos(data) ** 2
    run_benchmark(benchmark, m, compute_func, m.random.random, shape)