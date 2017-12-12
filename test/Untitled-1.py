import os
import pandas as pd
path = os.listdir('/home/antastsy/下载/dlcl/res')
df = pd.DataFrame()
for file in path:
    data = pd.read_csv('/home/antastsy/下载/dlcl/res/' + file)
    df = df.append(data, ignore_index=True)
df.to_csv('/home/antastsy/下载/dlcl/result.csv')