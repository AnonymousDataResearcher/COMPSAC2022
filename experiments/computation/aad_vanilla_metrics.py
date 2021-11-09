import pandas as pd

import strings
from common import vanilla_ize, vanilla_columns, evaluation_metrics
from experiments.management.locations_manager import SingleDatasetManager
from metric import Metric


# TODO: merge this with aac, that makes much more sense
def execute(slm: SingleDatasetManager, ca: str):
    if ca in [strings.VORONOI, strings.DBSCAN]:
        pass
    else:
        raise NotImplementedError(f'Not implemented for clustering algorithm {ca}')

    results = pd.DataFrame(columns=evaluation_metrics.keys(), index=Metric.metric_implementations(slm.metric))
    old_results = slm.import_raw(ca)
    if set(vanilla_columns(slm.real, ca)).issubset(old_results.columns):
        return

    def get_labels(mic_):
        fn = slm.fn_labels(ca, mic_)
        if not fn.exists():
            return None
        return pd.read_csv(fn).iloc[:, 0]

    for metric in slm.metric:
        mic_vanilla = Metric.vanilla_implementations(metric)
        vanilla_labels = get_labels(mic_vanilla)
        if vanilla_labels is None:
            continue

        for mic in Metric.metric_implementations(metric):
            mic_labels = get_labels(mic)
            if mic_labels is None:
                continue
            for k, v in evaluation_metrics.items():
                results.loc[mic, k] = v(vanilla_labels, mic_labels)

    results.columns = [vanilla_ize(x) for x in results.columns]
    results.index.name = strings.METRIC_IMPLEMENTATION_CODE

    new_results = old_results.join(results, how='outer')
    slm.export_raw(new_results, ca)
