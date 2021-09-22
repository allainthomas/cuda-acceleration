"""
To test : python3 -m pytest test_scripts/test_fft.py

With a Intel Xeon 2.20 GHz and a Tesla K80 :

--------------------------------------------------------------------------------------------------------- benchmark: 6 tests --------------------------------------------------------------------------------------------------------
Name (time in us)                     Min                       Max                      Mean                  StdDev                    Median                     IQR            Outliers         OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_FFT[shape0-cupy]            479.8100 (1.0)            686.3160 (1.0)            550.0186 (1.0)           83.4334 (1.0)            511.4690 (1.0)          104.3725 (1.0)           1;0  1,818.1203 (1.0)           5           1
test_FFT[shape0-numpy]        10,737.9530 (22.38)       14,492.1300 (21.12)       12,298.2842 (22.36)      1,366.5905 (16.38)       12,187.6650 (23.83)      1,226.5660 (11.75)         2;0     81.3122 (0.04)          5           1
test_FFT[shape1-cupy]         19,324.6860 (40.28)       20,174.2200 (29.39)       19,516.0886 (35.48)        368.3362 (4.41)        19,366.2340 (37.86)        229.2607 (2.20)          1;1     51.2398 (0.03)          5           1
test_FFT[shape2-cupy]         62,737.0390 (130.75)      68,826.0190 (100.28)      65,075.0552 (118.31)     2,618.3129 (31.38)       64,407.4130 (125.93)     4,360.9162 (41.78)         1;0     15.3669 (0.01)          5           1
test_FFT[shape1-numpy]       363,163.6530 (756.89)     465,188.3760 (677.80)     389,950.6638 (708.98)    42,392.3730 (508.10)     373,533.3340 (730.31)    31,078.3012 (297.76)        1;1      2.5644 (0.00)          5           1
test_FFT[shape2-numpy]     1,578,099.5730 (>1000.0)  1,994,339.4050 (>1000.0)  1,669,607.8368 (>1000.0)  181,763.8977 (>1000.0)  1,591,309.3850 (>1000.0)  118,931.9757 (>1000.0)       1;1      0.5989 (0.00)          5           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
=================== 6 passed, 2 warnings in 83.74s (0:01:23) ===================
  
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
def test_FFT(benchmark, module, shape):
    m = importlib.import_module(module)

    data_func = lambda shape: m.exp(2j * m.pi * m.random.random(shape))
    run_benchmark(benchmark, m, m.fft.fft, data_func, shape)