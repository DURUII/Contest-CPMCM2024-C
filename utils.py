""" 
DATA PRE-PROCESSING

AUTHOR: durui
DATE: 2024-09-21
"""


import pickle
import pandas as pd
import numpy as np


def general_process(input_filepath: str = 'dataset/附件一（训练集）.xlsx', output_filepath: str = 'data_general.pkl'):
    # merge 4 sheets into one, adding the sheet name as a column
    xls = pd.ExcelFile(input_filepath)
    sheet_names = ['材料1', '材料2', '材料3', '材料4']
    df_list = []
    for sheet in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        # hidden flaw in data
        if sheet == '材料2':
            df.rename(columns={'0（磁通密度，T）': '0（磁通密度B，T）'}, inplace=True)
        df['type_material'] = sheet.replace('材料', '')
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # rename the columns to English names
    column_mapping = {
        '温度，oC': 'temp',
        '频率，Hz': 'freq',
        '磁芯损耗，w/m3': 'core_loss',
        '0（磁通密度B，T）': 0,
        '励磁波形': 'type_waveform',
    }
    merged_df.rename(columns=column_mapping, inplace=True)

    # reorder the columns
    cols = ['type_material', 'temp', 'freq', 'core_loss', 'type_waveform'] + [i for i in range(1024)]
    merged_df = merged_df[cols]
    
    # convert columns to float in a vectorized manner
    merged_df.iloc[:, 5:] = merged_df.iloc[:, 5:].astype(float)

    # create a new column 'flux_density' by combining all columns
    # from index 5 onwards into a single array
    merged_df['flux_density'] = merged_df.iloc[:, 5:].values.tolist()
    merged_df.drop(columns=[i for i in range(1024)], inplace=True)
    
    assert len(merged_df.columns) == 6
    assert len(merged_df[merged_df['flux_density'].apply(lambda x: np.isnan(x).any())]) == 0 

    # save the dataframe using pickle for faster loading and processing
    with open(output_filepath, 'wb') as f:
        pickle.dump(merged_df, f)

if __name__ == "__main__":
    general_process()
