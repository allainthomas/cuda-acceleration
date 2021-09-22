"""
To test : python3 -m pytest test_scripts/test_svd.py

With a Intel Xeon 2.20 GHz and a Tesla K80

------------------------------------------------------------------------------------------------ benchmark: 4 tests -----------------------------------------------------------------------------------------------
Name (time in ms)                   Min                     Max                    Mean              StdDev                  Median                   IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_SVD[shape0-numpy]         702.1795 (1.0)          727.4271 (1.0)          713.9034 (1.0)       10.7720 (3.22)         712.8331 (1.0)         18.9686 (5.92)          2;0  1.4007 (1.0)           5           1
test_SVD[shape0-cupy]        1,241.5205 (1.77)       1,250.4484 (1.72)       1,247.0814 (1.75)       3.3435 (1.0)        1,247.7447 (1.75)         3.2062 (1.0)           2;0  0.8019 (0.57)          5           1
test_SVD[shape1-cupy]       73,409.6708 (104.55)    73,539.8227 (101.10)    73,473.6959 (102.92)    46.6552 (13.95)     73,475.3391 (103.08)      48.4527 (15.11)         2;0  0.0136 (0.01)          5           1
test_SVD[shape1-numpy]     101,872.5102 (145.08)   103,713.8633 (142.58)   102,983.3562 (144.25)   787.4558 (235.52)   103,075.1869 (144.60)   1,323.7676 (412.88)        1;0  0.0097 (0.01)          5           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
================== 4 passed, 1 warning in 1079.51s (0:17:59) ===================

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
@pytest.mark.parametrize("shape", [ (1000, 1000), (5000, 5000)])
#@pytest.mark.parametrize("shape", [(1000, 1000), (5000, 5000)]), (10000, 10000)])
def test_SVD(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.linalg.svd, m.random.random, shape)