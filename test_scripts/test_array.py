"""
To test : python3 -m pytest test_scripts/test_array.py

With a Intel Xeon 2.20 GHz and a Tesla K80

--------------------------------------------------------------------------------------------------------- benchmark: 6 tests --------------------------------------------------------------------------------------------------------
Name (time in us)                   Min                     Max                    Mean                 StdDev                  Median                    IQR            Outliers         OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_arr[shape0-cupy]           95.3560 (1.0)          362.4970 (1.0)          157.6850 (1.0)         115.1485 (1.0)          106.0300 (1.0)          88.2818 (1.0)           1;1  6,341.7573 (1.0)           5           1
test_arr[shape0-numpy]         788.5260 (8.27)       3,065.7710 (8.46)       1,306.5042 (8.29)        986.4789 (8.57)         890.6140 (8.40)        701.9367 (7.95)          1;1    765.4013 (0.12)          5           1
test_arr[shape1-cupy]        1,614.4430 (16.93)      1,970.3060 (5.44)       1,698.5158 (10.77)       152.3600 (1.32)       1,637.6340 (15.45)       102.6293 (1.16)          1;1    588.7493 (0.09)          5           1
test_arr[shape2-cupy]        6,344.4570 (66.53)      7,257.2000 (20.02)      6,538.7438 (41.47)       401.8222 (3.49)       6,367.2580 (60.05)       247.0783 (2.80)          1;1    152.9346 (0.02)          5           1
test_arr[shape1-numpy]      38,961.0250 (408.58)    74,126.0770 (204.49)    46,370.4428 (294.07)   15,521.9582 (134.80)    39,399.2400 (371.59)    9,467.8275 (107.25)        1;1     21.5655 (0.00)          5           1
test_arr[shape2-numpy]     154,138.7560 (>1000.0)  292,384.0240 (806.58)   182,906.2058 (>1000.0)  61,224.2961 (531.70)   155,172.4700 (>1000.0)  37,612.8098 (426.05)        1;1      5.4673 (0.00)          5           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
========================= 6 passed, 1 warning in 3.33s =========================

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
def test_arr(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.array, m.random.random, shape)