import pandas as pd
import time

import numpy as np
from competitors.aqbc_ops.jitted import AQBC as AQBC_JIT
from competitors.aqbc_ops.original import AQBC as AQBC_OG
from functions import confidence_interval


def compute(is_jit, x, n_bits=None, epochs=5, seed=0):
    if n_bits is None:
        n_bits = x.shape[1]

    if is_jit:
        a = AQBC_JIT(np.ascontiguousarray(x.T), n_bits, epochs, seed)
    else:
        a = AQBC_OG(x.T, n_bits, epochs, seed)

    a.optimize_all()
    a.hash(np.ascontiguousarray(x.T))
    return a.B


def run():
    rng = np.random.RandomState(684)
    idx_cols = ['iteration', 'nb', 'ep', 'sd']
    times = pd.DataFrame(columns=idx_cols + ['t_jit', 't_og']).set_index(idx_cols)

    for i in range(100):
        x_ = rng.rand(100 * (i + 1), 10)
        for nb in range(2, x_.shape[1] + 1, 3):
            for ep in range(1, 10, 3):
                for sd in range(1, 10, 3):
                    t0 = time.process_time()
                    ans_jit = compute(True, x_, n_bits=nb, seed=sd, epochs=ep)
                    t1 = time.process_time()
                    ans_og = compute(False, x_, n_bits=nb, seed=sd, epochs=ep)
                    t2 = time.process_time()

                    # Check equal
                    diff = np.abs(ans_jit - ans_og).max(axis=None)
                    assert np.array_equal(ans_jit, ans_og), f'unequal for {nb},{ep},{sd}: {diff}'

                    # Save durations
                    times.loc[(i, nb, ep, sd), 't_jit'] = t1 - t0
                    times.loc[(i, nb, ep, sd), 't_og'] = t2 - t1

    speedups = times.loc[times.t_jit != 0, 't_og'].divide(times.loc[times.t_jit != 0, 't_jit'])

    m, c = confidence_interval.get_mean_and_ci(speedups, .95)
    print(f'jitted is {m:.3f}+-{c:.3f} times faster based on {len(speedups) / len(times) * 100:.0f}% of experiments')
    print(f'mean of non-jitted on remainder of experiments = {times.loc[times.t_jit == 0, "t_og"].mean():.2f}')


if __name__ == '__main__':
    run()
