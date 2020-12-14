from datetime import datetime
import os
import pandas as pd

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
    {"place": "LaSalle St & Washington St", "id": 98},
    # {"place": "Dearborn St & Van Buren St", "id": 624},
    {"place": "State St & Van Buren St", "id": 33},
]


def load_dataset():
    df = load_parts()
    df["start_time"] = pd.to_datetime(
        df["start_time"], format='%Y-%m-%d %H:%M:%S')
    df = df.reset_index(drop=True).set_index("start_time")

    # Select only a few of them
    print(f"Loading only stations with the following id: {SELECTED_STATIONS}")
    df = df[df['from_station_id'].isin([s["id"] for s in SELECTED_STATIONS])]
    return df


def transform_time(df):
    df['hour'] = df.index.hour + 1  # 1 to 24
    df['day_of_month'] = df.index.day
    df['day_of_week'] = df.index.dayofweek + 1  # 1 to 7
    df['month'] = df.index.month
    df['year'] = df.index.year
    return df


def split_dataset(df, train_from=datetime(2014, 1, 1)):
    split_date = datetime(2018, 12, 31, 23, 59, 59)
    train_df = df[(df.index >= train_from) & (df.index <= split_date)]
    train_df = train_df.sort_values(by="start_time")
    train_df = transform_time(train_df)
    val_df = df.loc[df.index > split_date]
    val_df = val_df.sort_values(by="start_time")
    val_df = transform_time(val_df)
    print(
        f"Training from {str(train_df.iloc[0].name)} to {str(train_df.iloc[-1].name)}")
    print(
        f"Validating from {str(val_df.iloc[0].name)} to {str(val_df.iloc[-1].name)}")
    return train_df, val_df
