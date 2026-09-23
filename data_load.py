import pandas as pd
from pathlib import Path

PATH = Path(r'.\dataset')

dataset = pd.read_csv(PATH / 'okm_augumented_2021.csv')

