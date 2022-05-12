import pandas as pd
import gc

def downcast_df_int_columns(df):
    list_of_columns = list(df.select_dtypes(include=["int32", "int64"]).columns)

    if len(list_of_columns) >= 1:
        for col in list_of_columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    gc.collect()
    return df

def downcast_df_float_columns(df):
    list_of_columns = list(df.select_dtypes(include=["float64"]).columns)

    if len(list_of_columns) >= 1:
        for col in list_of_columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

    gc.collect()
    return df