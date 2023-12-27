# date
import pandas as pd

days = pd.date_range('2020-01-01', '2022-01-01', freq='D')
for idx in range(len(days)-1):
    start = days[idx]
    end = days[idx+1]
    print(str(start)[:16].replace(' ', 'T'))