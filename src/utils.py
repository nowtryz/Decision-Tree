from pandas import DataFrame


def pd_index(df: DataFrame, name: str):
    return df.columns.get_loc(name)
