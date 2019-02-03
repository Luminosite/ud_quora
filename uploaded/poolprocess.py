from multiprocessing import Pool
import time
import pandas as pd
import numpy as np


def process_data(func, data, n, tag, data_n=-1):
    n = n if data_n == -1 or data_n >= n else data_n
    data_block_num = n if data_n == -1 else data_n
    print("process %s starts with %d processes %d jobs" % (tag, n, data_block_num))

    t_start = time.time()
    l = data.shape[0]
    block_len = int((l-1)/n) + 1

    data_list = []
    for i in range(data_block_num):
        part = data[i*block_len: (i+1)*block_len].copy()
        data_list.append(part)

    pool = Pool(processes=n)
    ret = pool.map(func, data_list)
    pool.close()
    pool.join()

    result = pd.concat(ret, axis=0).reset_index(drop=True)

    t_end = time.time()
    print("process %s finished in %f s" % (tag, (t_end-t_start)))
    return result


def test_func(data):
    time.sleep(3)
    return data*(-1)


if __name__ == '__main__':
    for _ in range(10):
        d = pd.DataFrame(np.array(range(29)))
        d = process_data(test_func, d, 5, "test func")
        print(d)
