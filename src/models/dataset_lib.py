import os
import pandas as pd


def load_dataset():
    data_folder = f"{os.getcwd()}/../../data"
    output_file = f"{data_folder}/trips.csv"
    with open(output_file, 'wb') as output:
        for part in range(1, 15):
            with open(f"{data_folder}/parts/chicago-divvy-trips-part-{part}", 'rb') as input_file:
                print(
                    f"Putting together {data_folder}/parts/chicago-divvy-trips-part-{part}")
                output.write(input_file.read())
    df = pd.read_csv(output_file)
    df["start_time"] = pd.to_datetime(
        df["start_time"], format='%Y-%m-%d %H:%M:%S')
    df = df.reset_index(drop=True).set_index("start_time")
    return df


def split_dataset(df):
    split_date = pd.datetime(2018, 12, 31, 23, 59, 59)
    train_df = df.loc[df.index <= split_date]
    train_df = train_df.sort_values(by="start_time")
    val_df = df.loc[df.index > split_date]
    val_df = val_df.sort_values(by="start_time")
    print(
        f"Training from {str(train_df.iloc[0].name)} to {str(train_df.iloc[-1].name)}")
    print(
        f"Validating from {str(val_df.iloc[0].name)} to {str(val_df.iloc[-1].name)}")
    return train_df, val_df
