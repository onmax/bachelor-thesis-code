from datetime import date, datetime
import os
from os.path import join
import pandas as pd
import numpy as np

import os
import sys
import inspect
preprocessing = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))) + "../preprocessing/"
sys.path.insert(1, os.path.join(sys.path[0], '../preprocessing'))
if True:
    from preprocessing_lib import load_parts


SELECTED_STATIONS = [
    {"place": "Aquarium", "id": 3},
    {"place": "Train station", "id": 77},
    {"place": "Nightlife Neighborhood", "id": 106},
    {"place": "Shops", "id": 117},
    {"place": "Office district", "id": 199},
    {"place": "Suburban neighborhood", "id": 299}
]

SIMILAR_STATIONS = [
    # {"place": "LaSalle St & Jackson Blvd", "id": 283},
    # {"place": "Franklin St & Jackson Blvd", "id": 36},
    # {"place": "LaSalle St & Adams St", "id": 40},
    # {"place": "Franklin St & Monroe St", "id": 287},
    {"place": "Dearborn St & Adams St", "id": 37},
    # {"place": "Clark St & Ida B Wells Dr", "id": 50},
    # {"place": "Dearborn St & Monroe St", "id": 49},
    # {"place": "LaSalle St & Washington St", "id": 98},
    # {"place": "Dearborn St & Van Buren St", "id": 624},
    # {"place": "State St & Van Buren St", "id": 33},
]

SELECTED_STATIONS = SIMILAR_STATIONS

features = ["hour", "day_of_week", "month"]


def load_dataset(base_path=""):
    df = load_parts(base_path)
    df["start_time"] = pd.to_datetime(
        df["start_time"], format='%Y-%m-%d %H:%M:%S')
    df = df.reset_index(drop=True).set_index("start_time")
    return df


def transform_time(df):
    if "hour" in features:
        df['hour'] = df.index.hour  # 0 to 23
    if "day_of_month" in features:
        df['day_of_month'] = df.index.day
    if "day_of_week" in features:
        df['day_of_week'] = df.index.dayofweek + 1  # 1 to 7
    if "month" in features:
        df['month'] = df.index.month
    if "year" in features:
        df['year'] = df.index.year
    return df


def small_dataset(df, from_date, to_date):
    sub_df = df[(df.index >= from_date) & (df.index <= to_date)]
    sub_df = sub_df.sort_values(by="start_time")
    sub_df = transform_time(sub_df)
    return sub_df


def split_dataset(df, train_year=2017, validation_year=2018, test_year=2019):
    train_from = datetime(train_year, 1, 1, 0)
    train_to = datetime(train_year, 12, 31, 23, 59, 59)
    train_df = small_dataset(df, train_from, train_to)
    print(
        f"Training from {str(train_df.iloc[0].name)} to {str(train_df.iloc[-1].name)}")

    val_from = datetime(validation_year, 1, 1, 0)
    val_to = datetime(validation_year, 12, 31, 23, 59, 59)
    val_df = small_dataset(df, val_from, val_to)
    print(
        f"Validating from {str(val_df.iloc[0].name)} to {str(val_df.iloc[-1].name)}")

    test_from = datetime(test_year, 1, 1, 0)
    test_to = datetime(test_year, 12, 31, 23, 59, 59)
    test_df = small_dataset(df, test_from, test_to)
    print(
        f"Testing from {str(test_df.iloc[0].name)} to {str(test_df.iloc[-1].name)}")

    return train_df, val_df, test_df


def divide_X_Y(df):

    X = df[features]
    Y = df.drop(columns=features)
    return X, Y


def remove_first_day(df):
    df = df[df.index >= df.index.min().date()+pd.offsets.Day(7)]
    return df


def shift(df):
    q_columns = df.columns.drop(["hour", 'day_of_week', 'month'])
    df_shifted = df[q_columns].shift(7, freq='D')
    df_shifted.drop(df_shifted[df_shifted.index >= df_shifted.index.max().date()-pd.offsets.Day(0)].index, inplace=True)
    df_shifted = df_shifted[df_shifted.index >= df_shifted.index.min().date()+pd.offsets.Day(0)]
    df = remove_first_day(df)
    df_final = df[["hour", "day_of_week", "month"]].join(df_shifted, how="right")
    df_final = df_final.dropna(0)
    df_final["start_time"] = df.index
    df_final = df_final.reset_index(drop=True).set_index("start_time")
    return df_final