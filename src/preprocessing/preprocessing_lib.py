import os
import pandas as pd


def path():
    current_path = os.getcwd()
    return f"{current_path}/../../data"


def data_path(year):
    return f"{path()}/{year}"


def get_csvs(years):
    current_path = path()
    paths = []
    for year in years:
        pickle_path = f"{current_path}/{year}/trips-{year}.csv"
        paths.append(pickle_path)
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


def merge_years(inputs):
    df = pd.DataFrame()
    for input in inputs:
        print(f"Reading {input}")
        df_temp = pd.read_csv(input)
        df = pd.concat([df, df_temp], join='outer')
    df['start_time'] = df['start_time'].combine_first(df['starttime'])
    df = df.drop(columns=["starttime"])
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


def load_parts():
    parts = [
        f"{path()}/parts/chicago-divvy-trips-part-{i}" for i in list(range(1, 15))]
    output_file = f'{path()}/trips.csv'
    join(output_file, parts)
    df = pd.read_csv(output_file)
    return df
