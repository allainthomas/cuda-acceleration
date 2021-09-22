"""
To test : python3 -m pytest test_scripts/test_sum.py

With a Intel Xeon 2.20 GHz and a Tesla K80

-------------------------------------------------------------------------------------------------- benchmark: 6 tests --------------------------------------------------------------------------------------------------
Name (time in us)                  Min                     Max                   Mean                 StdDev                 Median                    IQR            Outliers         OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_Sum[shape0-numpy]        424.1620 (1.0)          537.4680 (1.0)         457.0702 (1.0)          46.5005 (1.60)        443.8630 (1.0)          47.8940 (1.62)          1;0  2,187.8477 (1.0)           5           1
test_Sum[shape0-cupy]       1,190.7270 (2.81)       1,268.4390 (2.36)      1,217.4516 (2.66)         33.4022 (1.15)      1,197.4030 (2.70)         48.2437 (1.63)          1;0    821.3879 (0.38)          5           1
test_Sum[shape1-numpy]     16,579.5990 (39.09)     16,865.4120 (31.38)    16,731.4914 (36.61)       128.7438 (4.43)     16,720.0180 (37.67)       238.2457 (8.04)          2;0     59.7675 (0.03)          5           1
test_Sum[shape1-cupy]      28,876.9680 (68.08)     28,949.5860 (53.86)    28,899.6048 (63.23)        29.0885 (1.0)      28,892.7380 (65.09)        29.6370 (1.0)           1;0     34.6025 (0.02)          5           1
test_Sum[shape2-numpy]     66,365.4590 (156.46)    68,129.0320 (126.76)   66,977.3392 (146.54)      780.0045 (26.81)    66,541.3500 (149.91)    1,218.4070 (41.11)         1;0     14.9304 (0.01)          5           1
test_Sum[shape2-cupy]      66,850.3020 (157.61)   102,607.8230 (190.91)   78,626.5888 (172.02)   16,637.8764 (571.97)   67,082.5740 (151.13)   26,042.4230 (878.71)        1;0     12.7183 (0.01)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
========================= 6 passed, 1 warning in 3.22s =========================


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
def test_Sum(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.sum, m.random.random, shape)