import pandas as pd

import strings as s
from common import summary_file, data_folder


def get_summary():
    # Returns summary with properties of real datasets
    return pd.read_csv(summary_file).set_index(s.DATASET)


def get_properties(dataset_name):
    if isinstance(dataset_name, str):
        if is_real_dataset(dataset_name):
            return get_summary().loc[dataset_name]
        else:
            shape, ndp, nf, k, seed, *additional = dataset_name.split('_')
            sr = pd.Series(data={s.NUMBER_OF_DATAPOINTS: int(ndp),
                                 s.NUMBER_OF_FEATURES: int(nf),
                                 s.NUMBER_OF_CLASSES: int(k),
                                 s.ABBREVIATION: None,
                                 s.ANNOTATED: True},
                           name=dataset_name)
            if len(additional) == 0:
                sr[s.RADIUS] = 1.0
            elif len(additional) == 1:
                sr[s.RADIUS] = float(additional[0])
            else:
                raise NotImplementedError(f'Not implemented for {len(additional)} additional arguments')
            return sr

    elif isinstance(dataset_name, pd.Series):
        if all([is_real_dataset(dsn) for dsn in dataset_name.to_numpy()]):
            return dataset_name.to_frame().merge(right=get_summary(), left_on=s.DATASET, right_index=True) \
                .set_index(s.DATASET).assign(**{s.SHAPE: pd.NA, s.SEED: pd.NA, s.RADIUS: pd.NA})
        elif all([dsn.startswith(s.BLOB) or dsn.startswith(s.SPARSE_BLOB) for dsn in dataset_name.to_numpy()]):
            df_info = dataset_name.str.split('_', expand=True)
            df = pd.DataFrame(data={s.DATASET: dataset_name,
                                    s.SHAPE: df_info.iloc[:, 0],
                                    s.NUMBER_OF_DATAPOINTS: df_info.iloc[:, 1].astype(int),
                                    s.NUMBER_OF_FEATURES: df_info.iloc[:, 2].astype(int),
                                    s.NUMBER_OF_CLASSES: df_info.iloc[:, 3].astype(int),
                                    s.SEED: df_info.iloc[:, 4].astype(int),
                                    s.ABBREVIATION: None,
                                    s.ANNOTATED: True}).set_index(s.DATASET)
            if len(df_info.columns) == 5:
                df.loc[:, s.RADIUS] = 1.0
            elif len(df_info.columns) == 6:
                df.loc[:, s.RADIUS] = df_info.iloc[:, 5].astype(float).fillna(1.0).to_numpy()
            else:
                raise NotImplementedError(f'Not implemented for {len(df_info.columns)} arguments')
            return df
        else:
            raise NotImplementedError('Not implemented for dataset names')
    raise NotImplementedError(f'Not implemented for type : {type(dataset_name)}')


def add_properties(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    assert isinstance(df, pd.DataFrame)
    if s.DATASET in df.columns:
        all_datasets = df[s.DATASET].drop_duplicates()
        df_prop = get_properties(all_datasets)
        return df.merge(df_prop, left_on=s.DATASET, right_index=True, how='left')
    elif df.index.name == s.DATASET:
        all_datasets = df.reset_index()[s.DATASET].drop_duplicates()
        df_prop = get_properties(all_datasets)
        return df.merge(df_prop, left_index=True, right_index=True, how='left')
    else:
        raise NotImplementedError()


def print_tex():
    df = get_summary().sort_values(s.NUMBER_OF_DATAPOINTS).iloc[2:-3].reset_index()
    d = {s.ABBREVIATION: '', s.DATASET: 'Name', s.NUMBER_OF_DATAPOINTS: '$|D|$', s.NUMBER_OF_FEATURES: '$|F|$',
         s.NUMBER_OF_CLASSES: '$k$', }
    df[s.DATASET] = df[s.DATASET].str.replace('_', ' ').str.capitalize()
    df = df.rename(columns=d)[list(d.values())]
    print(df.to_latex(escape=False, index=False))


if __name__ == '__main__':
    print_tex()


def is_valid_dataset_name(dataset_name):
    if dataset_name.split('_')[0] in s.all_shapes:
        return True
    else:
        return get_dataset_fn(dataset_name).exists()


def is_real_dataset(dataset_name):
    return dataset_name in get_summary().index


def get_dataset_fn(dataset_name):
    return data_folder / f'{dataset_name}.csv'
