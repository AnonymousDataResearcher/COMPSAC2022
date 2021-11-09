import string
import sys
import warnings

import pandas as pd
from pandas.errors import DtypeWarning
from sklearn.datasets import load_iris

from common import raw_data_folder, data_folder, summary_file
from strings import LABEL

data_folder.mkdir(exist_ok=True, parents=True)


# Each of the following methods assume that the files as specified have been downloaded to bitbooster/data/raw_data


def prep_localization():
    """
    Data gathered from https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity
    See Data Set Description there for details of tag_dict
    """
    fn_in = raw_data_folder / 'ConfLongDemo_JSI.txt'
    fn_out = data_folder / 'localization.csv'

    df = pd.read_csv(fn_in, header=None)
    df.columns = ['person', 'tag_id', 'timestamp', 'date'] + list('xyz') + [LABEL]
    tag_dict = {
        "010-000-024-033": "ANKLE_LEFT",
        "010-000-030-096": "ANKLE_RIGHT",
        "020-000-033-111": "CHEST",
        "020-000-032-221": "BELT"
    }

    # Full set
    df.iloc[:, -4:].to_csv(fn_out, index=False)

    # Set per tag
    for tag_id, df_tag in df.groupby('tag_id'):
        df_tag.iloc[:, -4:].to_csv(data_folder / f'localization_{tag_dict[tag_id]}.csv', index=False)


def prep_miniboone():
    """
    Data gathered from https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification
    Keep only MiniBooNE_PID.txt
    """
    fn_in = raw_data_folder / 'MiniBooNE_PID.txt'
    fn_temp = raw_data_folder / 'temp.csv'
    fn_out = data_folder / 'miniboone.csv'

    with open(fn_in, 'r') as rf:
        class_1_size, class_2_size = [int(x) for x in rf.readline().strip().split(' ')]
        with open(fn_temp, 'w+') as wf:
            for line in rf.readlines():
                wf.write(line[2:].replace('  ', ',').replace(' -', ','))

    df = pd.read_csv(fn_temp, header=None)
    assert len(df) == class_1_size + class_2_size
    df.columns = [f'feature{x}' for x in range(1, len(df.columns) + 1)]
    df.loc[0:class_1_size, LABEL] = 0
    df.loc[class_1_size:, LABEL] = 1
    df.to_csv(fn_out, index=False)
    fn_temp.unlink()


def prep_skin():
    """
    Data gathered from https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
    Keep only Skin_NonSkin.txt
    """

    fn_in = raw_data_folder / 'Skin_NonSkin.txt'
    fn_out = data_folder / 'skin.csv'
    fn_out2 = data_folder / 'skin_filtered.csv'

    df = pd.read_csv(fn_in, sep='\t', header=None)
    df.columns = list('BGR') + [LABEL]
    df.to_csv(fn_out, index=False)

    # Save a separate copy that does not have 0-vectors as datapoints, because JKKC can't handle them
    df[~(df.drop(columns=LABEL) == 0).all(axis=1)].to_csv(fn_out2, index=False)


def prep_postures():
    """
    Data gathered from https://archive.ics.uci.edu/ml/datasets/Motion+Capture+Hand+Postures
    Extracted zip contains 1 file Postures.csv
    """
    fn_in = raw_data_folder / 'Postures.csv'
    fn_out = data_folder / 'postures.csv'
    df = pd.read_csv(fn_in, skiprows=[1]).drop(columns='User').rename(columns={'Class': LABEL})

    # Remove some columns. This makes sure that about 83% of the data has no missing values
    drop_cols = sum([[f'X{i}', f'Y{i}', f'Z{i}'] for i in range(6, 12)], [])
    df = df.drop(columns=drop_cols)

    # Drop datapoints with missing values
    df = df[~(df == '?').any(axis=1)]
    df.to_csv(fn_out, index=False)


def prep_credit_card_default():
    """
    Data gathered from https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    Keep only 'default of credit card clients.xls'
    """
    df = pd.read_excel(raw_data_folder / 'default of credit card clients.xls', skiprows=[0])
    fn_out = data_folder / 'credit_card_default.csv'

    # apply label, drop ID
    df = df.rename(columns={'default payment next month': 'label'}).drop(columns='ID')

    # convert binary values
    df['SEX_MALE'] = df['SEX'].replace({2, 0})
    del df['SEX']

    # Convert categorical values
    categorical_values = {
        'EDUCATION': {1: 'GRADUATE_SCHOOL', 2: 'UNIVERSITY', 3: 'HIGH SCHOOL', 4: 'OTHERS'},
        'MARRIAGE': {1: 'MARRIED', 2: 'SINGLE', 3: 'OTHERS'},
    }
    for k, v in categorical_values.items():
        for k1, v1 in v.items():
            df[f'{k}_{v1}'] = df[k].apply(lambda x: 1 if x == k1 else 0)
        del df[k]

    df.to_csv(fn_out, index=False)


def prep_avila():
    """
    Data gathered from https://archive.ics.uci.edu/ml/datasets/Avila
    zip contains 3 files, keep only avila-tr.txt and avila-ts.txt
    """
    fn_in1 = raw_data_folder / 'avila-tr.txt'
    fn_in2 = raw_data_folder / 'avila-ts.txt'
    fn_out = data_folder / 'avila.csv'
    df1 = pd.read_csv(fn_in1, header=None)
    df2 = pd.read_csv(fn_in2, header=None)
    df = df1.append(df2)
    df.columns = [f'F{i}' for i in range(1, 11)] + [LABEL]
    df.to_csv(fn_out, index=False)


def prep_letter_recognition():
    """
    Data gathered from https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    uses letter-recognition.data only
    """
    fn_in = raw_data_folder / 'letter-recognition.data'
    fn_out = data_folder / 'letter_recognition.csv'
    df = pd.read_csv(fn_in, header=None)
    df.columns = [LABEL] + [f'F{i}' for i in range(1, len(df.columns))]
    df.to_csv(fn_out, index=False)


def prep_magic_gamma_telescope():
    """
    Data gathered from https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
    Use magic04.data only
    """
    fn_in = raw_data_folder / 'magic04.data'
    fn_out = data_folder / 'magic_gamma_telescope.csv'
    df = pd.read_csv(fn_in, header=None)
    df.columns = [f'F{i}' for i in range(1, len(df.columns))] + [LABEL]
    df.to_csv(fn_out, index=False)


def prep_eeg_eye_state():
    """
    Data gathered from https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
    Keep 'EEG Eye State.arff
    """
    fn_in = raw_data_folder / 'EEG Eye State.arff'
    fn_out = data_folder / 'eeg_eye_state.csv'
    df = pd.read_csv(fn_in, skiprows=19, header=None)
    df.columns = [f'F{i}' for i in range(1, len(df.columns))] + [LABEL]
    df.to_csv(fn_out, index=False)


def prep_pendigits():
    """
    https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
    Use files pendigits.tes, pendigits.tra
    """
    fn1 = raw_data_folder / 'pendigits.tes'
    fn2 = raw_data_folder / 'pendigits.tra'
    fn_out = data_folder / 'pendigits.csv'
    df1 = pd.read_csv(fn1, sep=',', header=None)
    df2 = pd.read_csv(fn2, sep=',', header=None)
    df = df1.append(df2)
    df.columns = sum([[f'X{i}', f'Y{i}'] for i in range(1, 9)], []) + [LABEL]
    df = df.applymap(lambda x: str(x).strip())
    df.to_csv(fn_out, index=False)


def prep_polish_bankruptcy():
    """
    https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
    Use all 5 files
    Datapoints with missing values are removed
    """
    fn_out = data_folder / 'polish_bankruptcy.csv'

    df = pd.DataFrame()
    warnings.filterwarnings(action='ignore', category=DtypeWarning)
    for i in range(1, 6):
        fn_in = raw_data_folder / f'{i}year.arff'

        df_i = pd.read_csv(fn_in, skiprows=69, header=None)
        df = df.append(df_i)
    warnings.resetwarnings()
    df.columns = [f'F{i}' for i in range(1, 65)] + [LABEL]

    # remove unknown values
    # noinspection PyUnresolvedReferences
    df = df[~(df == '?').any(axis=1)]
    df.to_csv(fn_out, index=False)


def prep_iris():
    """
    Toy dataset based on the Iris dataset available in sklearn
    """
    fn_out = data_folder / 'iris.csv'
    load_iris(as_frame=True)['frame'].rename(columns={'target': LABEL}).to_csv(fn_out, index=False)


def prep_electrical_grid_stability():
    """
    https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data
    See page for details. Use file Data_for_UCI_named.csv
    """
    fn_in = raw_data_folder / 'Data_for_UCI_named.csv'
    fn_out = data_folder / 'electrical_grid_stability.csv'
    pd.read_csv(fn_in).drop(columns=['stab', 'p1']).rename(columns={'stabf': LABEL}).to_csv(fn_out, index=False)


def prep_vicon():
    """
    https://archive.ics.uci.edu/ml/datasets/Vicon+Physical+Action+Data+Set
    See page for details
    The rar is extracted into a folder 'Vicon Physical Action Data Set', which contains 10 sub folders and a readme
    The entire folder should be put in bitbooster/data/raw_data as separate folder

    The created dataset is a classification for sub1 between punch and handshake (akin to Beer's Grace paper)
    """
    fd_sub1 = raw_data_folder / 'Vicon Physical Action Data Set' / 'sub1'
    fn1 = fd_sub1 / 'aggressive' / 'Punching.txt'
    fn2 = fd_sub1 / 'normal' / 'Handshaking.txt'

    df1 = pd.read_csv(fn1, sep=' ', header=None).dropna(axis=1).drop(columns=0).assign(**{LABEL: 'PUNCH'})
    df2 = pd.read_csv(fn2, sep=' ', header=None).dropna(axis=1).drop(columns=0).assign(**{LABEL: 'HANDSHAKE'})
    df = df1.append(df2)
    df.columns = [f'feature_{i:02}' for i in range(27)] + [LABEL]
    df.to_csv(data_folder / 'vicon1.csv', index=False)


def recompute_summary_file():
    """
    Computes the summary that contains the information of each dataset in a csv, basically a loopup table for easy
    access
    """
    from os import listdir
    from os.path import isfile
    import strings as s

    # Gather all known datasets
    datasets = [f[:-4] for f in listdir(data_folder) if isfile(data_folder / f)]
    df = pd.DataFrame(index=datasets)

    # Extract info for each dataset
    for ds in datasets:
        data = pd.read_csv(data_folder / f'{ds}.csv')
        df.loc[ds, s.ANNOTATED] = LABEL in data.columns
        df.loc[ds, s.NUMBER_OF_FEATURES] = len(data.columns) - df.loc[ds, s.ANNOTATED]
        df.loc[ds, s.NUMBER_OF_DATAPOINTS] = len(data)
        df.loc[ds, s.NUMBER_OF_CLASSES] = pd.NA if not df.loc[ds, s.ANNOTATED] else len(data[LABEL].unique())
    df = df.sort_values(s.NUMBER_OF_DATAPOINTS)
    df[s.ABBREVIATION] = list((string.ascii_uppercase + '0123456789' + string.ascii_lowercase)[:len(df)])

    # Naming / Typing
    df.index.name = s.DATASET
    for x in [s.NUMBER_OF_FEATURES, s.NUMBER_OF_DATAPOINTS, s.NUMBER_OF_CLASSES]:
        df.loc[:, x] = df.loc[:, x].astype(int)

    # Saving
    df.to_csv(summary_file)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        recompute_summary_file()
    elif len(sys.argv) == 2 and sys.argv[1] == 'all':
        data_folder.mkdir(exist_ok=True, parents=True)
        z = list(globals().keys())
        for method_name in z:
            try:
                if method_name.startswith('prep_'):
                    globals()[method_name]()
                    print(f'Dataset {method_name.replace("prep_", "")} has been processed')
            except FileExistsError:
                print(f'Raw data for {method_name.replace("prep_", "")} downloaded, skipping it.')
        recompute_summary_file()
    else:
        print('use preprocessing.py:\n'
              '[Without argument] to recompute summary file\n'
              'with argument "all" to preprocess all downloaded datasets')
