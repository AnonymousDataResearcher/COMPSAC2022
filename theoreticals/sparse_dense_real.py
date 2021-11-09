import numpy as np
import pandas as pd

import strings as s
from data.summary import get_summary
from functions.progress import ProgressShower

# Get all average 4 dist values
sr = pd.Series(dtype=float)
for fn in ProgressShower('results/SAC_REAL_NORM/4dist'):
    if fn.name == '_eps.csv' or fn.name == '_avg.csv' or fn.name == '_avg_4dist.csv':
        continue
    sr[fn.name[:-4]] = pd.read_csv(fn)['EVAN'].mean()

sr.index.name = s.DATASET
sr = sr.divide(np.power(get_summary()[s.NUMBER_OF_FEATURES], 0.5)).sort_values()
sr.name = '4dist'

sr.to_csv('results/SAC_REAL_NORM/4dist/_avg_4dist.csv')

# Sort on value of 4dist / sqrt(|F|)
print(sr.to_string())
