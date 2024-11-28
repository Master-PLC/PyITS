import pandas as pd


def shift_data(data_raw: pd.DataFrame, columns, shift=0):
    if shift == 0 or len(columns) == 0:
        return data_raw

    data = data_raw.copy()
    for col in columns:
        for i in range(1, shift + 1):
            data[f'{col}_{i}'] = data[col].shift(i)
    shifted_columns = [f'{col}_{i}' for col in columns for i in range(1, shift + 1)]
    data = data.iloc[shift:].reset_index(drop=True)
    return data, shifted_columns