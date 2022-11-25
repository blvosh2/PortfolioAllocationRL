import pandas as pd


def create_data(Path_to_csv, start_price, output_file_name):
    df = pd.read_csv(Path_to_csv)
    df['Price'] = 0
    df.at[df.index.stop - 1, 'Price'] = start_price

    for idx in reversed(range(df.index.stop - 1)):
        df.at[idx, 'Price'] = (df.at[idx + 1, 'Price'] / (float(df.at[idx + 1, 'Change'][:-1]) / 100 + 1))

    df.Change = df.Change.apply(lambda x: float(x[:-1]))
    df.to_csv(f'./data/{output_file_name}.csv')
    pass


if __name__ == '__main__':
    # UPRO preprocessing
    output_file_name = 'UPROSIM_preprocessed'
    Path_to_csv = './data/UPROSIM.csv'
    start_price = 23.28
    create_data(Path_to_csv=Path_to_csv, start_price=start_price, output_file_name=output_file_name)
    # TMF preprocessing
    output_file_name = 'TMFSIM_preprocessed'
    Path_to_csv = './data/TMFSIM.csv'
    start_price = 19.02
    create_data(Path_to_csv=Path_to_csv, start_price=start_price, output_file_name=output_file_name)