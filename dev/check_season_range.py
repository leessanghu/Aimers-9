import pandas as pd
raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['season'])
print(sorted(raw['season'].unique().tolist()))
