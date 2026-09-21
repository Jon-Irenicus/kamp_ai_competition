import pandas as pd
from pathlib import Path

PATH = Path(r'C:\Users\82105\kamp_ai_competition\dataset')

scaled_data = pd.read_csv(PATH / 'scaled_data.csv', encoding='cp949')
raw_data = pd.read_excel(PATH / 'Welding Data Set_01.xlsx', sheet_name=0)
result = pd.read_excel(PATH / 'Welding Data Set_01.xlsx', sheet_name=1)
result = result.drop([result.columns[-1]], axis=1)
data_info = pd.read_excel(PATH / 'Welding Data Set_01.xlsx', sheet_name=2)
