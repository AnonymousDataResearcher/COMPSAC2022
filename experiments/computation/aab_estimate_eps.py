import pandas as pd

import metric
import strings as s
from bitbooster.other.eps_estimation import estimate_eps
from experiments.management.locations_manager import SingleDatasetManager


def execute(slm):
    assert isinstance(slm, SingleDatasetManager)
    df = slm.import_4dist()
    sr_eps = pd.Series(name=s.EPS, dtype=float)
    sr_eps.index.name = s.METRIC_IMPLEMENTATION_CODE
    prev_eps = slm.import_eps()

    for c in df.columns:

        # Get current value
        if c in prev_eps.index:
            sr_eps[c] = prev_eps.loc[c]
            continue
            # current_guess = prev_eps.loc[c]
        elif c == metric.BITBOOSTED_AQBC:
            if metric.AQBC in sr_eps.index:
                # Already done in this round
                sr_eps[c] = sr_eps[metric.AQBC]
                continue
            elif metric.AQBC in prev_eps.index:
                # Already done previously
                sr_eps[c] = prev_eps.loc[c]
                continue
            else:
                # Not done yet do now
                current_guess = None
        else:
            current_guess = None

        sr_eps[c] = estimate_eps(n_dist_values=df[c], current_guess=current_guess, title=f'{c}/{slm.dataset_name}')

    slm.export_eps(sr_eps)
