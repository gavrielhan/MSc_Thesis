# config_local.py
from LabData import config_global
import pandas as pd
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader

# Set base to your desired base directory
base = '/Users/gavrielhannuna/PycharmProjects/MSc_Thesis/base'

# Override qc_dir to a writable location (for example, inside your MSc_Thesis folder)
qc_dir = '/Users/gavrielhannuna/PycharmProjects/MSc_Thesis/qc'

config_global.base = base
config_global.qc_dir = qc_dir


def load_body_measures_data(study_ids):
    loader = BodyMeasuresLoader()
    data_container = loader.get_data(study_ids=study_ids)

    if data_container is None or data_container.df is None:
        print("Warning: Failed to load data. The returned data is None.")
        return pd.DataFrame()  # Return an empty DataFrame or handle as needed

    return data_container.df

study_ids = [10, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
df = load_body_measures_data(study_ids=study_ids)

if df.empty:
    print("No data available for the provided study IDs.")
else:
    print(df.head(3))
