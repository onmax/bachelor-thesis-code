import os
import pandas as pd


def path(base_path=""):
    current_path = os.getcwd()
    return f"{current_path}{base_path}/../../data/files"


def path_data_for_year(year):
    return f"{path()}/{year}"


def get_csvs(years):
    current_path = path()
    paths = []
    for year in years:
        csv_path = f"{current_path}/{year}/trips-{year}.csv"
        paths.append(csv_path)
    return paths


def merge_csv(inputs, output, year, cols):
    df = pd.DataFrame()
    for input in inputs:
        print(f"Reading {input}")
        df_temp = pd.read_csv(input)
        df_temp[cols[0]] = pd.to_datetime(
            df_temp[cols[0]], format='%Y-%m-%d %H:%M:%S', infer_datetime_format=True)
        df = pd.concat([df, df_temp], join='outer')
    output_file = f"{output}/trips-{year}.csv"
    print(f"Writing {output_file}")
    df[cols].to_csv(output_file, index=False)


def merge_years(inputs, with_starttime=True):
    df = pd.DataFrame()
    for input in inputs:
        print(f"Reading {input}")
        df_temp = pd.read_csv(input)
        df = pd.concat([df, df_temp], join='outer')
    if with_starttime:
        df['start_time'] = df['start_time'].combine_first(df['starttime'])
        df = df.drop(columns=["starttime"])
    return df


def group_by_time_n_quantity(df):
    df["start_time"] = pd.to_datetime(
        df["start_time"], format='%Y-%m-%d %H:%M:%S')
    INTERVAL = "1H"  # It could be also 15Min
    df = df.groupby('from_station_id').resample(INTERVAL, on='start_time') \
        .size() \
        .to_frame() \
        .rename(columns={0: "quantity"}) \
        .reset_index() \
        .set_index("start_time")
    return df


def multiple_columns(df):
    df["quantity_index"] = "quantity_" + df["from_station_id"].astype("str")
    df = df.drop(columns=["from_station_id"])
    df = df.pivot(columns='quantity_index', values='quantity')
    df.columns.name = None
    return df.fillna(0)


def nn_format(df):
    df = group_by_time_n_quantity(df)
    df = multiple_columns(df)
    return df


def save_with_split(df, dest_folder, write_size):
    # Make a destination folder if it doesn't exist yet
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    else:
        # Otherwise clean out all files in the destination folder
        for file in os.listdir(dest_folder):
            os.remove(os.path.join(dest_folder, file))
    part_num = 0

    df.to_csv(f"{path()}/group.csv")

    with open(f"{path()}/group.csv", 'rb') as input:
        while True:
            chunk = input.read(write_size)
            if not chunk:
                # End the loop if we have hit EOF
                break
            part_num += 1

            print(f"Writing {dest_folder}/chicago-divvy-trips-part-{part_num}")
            # Create a new file name
            with open(f"{dest_folder}/chicago-divvy-trips-part-{part_num}", 'wb') as fd:
                fd.write(chunk)


def join(output_file, parts):
    with open(output_file, 'wb') as output:
        for part in parts:
            with open(part, 'rb') as input_file:
                print(f"Putting together {part}")
                output.write(input_file.read())


def load_parts(base_path=""):
    parts = [
        f"{path(base_path)}/../parts/chicago-divvy-trips-part-{i}" for i in list(range(1, 4))]
    output_file = f'{path(base_path)}/trips.csv'
    join(output_file, parts)
    df = pd.read_csv(output_file)
    return df
