# config_local.py
import pandas as pd
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.ItamarSleepLoader import ItamarSleepLoader

study_ids = [10, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]

bm =ItamarSleepLoader().get_data(study_ids=study_ids)
df = bm.df
metadata = bm.df_metadata
age_gender = metadata[['age', 'gender', 'yob', 'StudyTypeID']]
print(df.head(3))
print(df.columns)
print(df.shape)
print(metadata.head(3))
print(age_gender.head(3))