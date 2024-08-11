import pandas as pd


def csv_to_dict(data):

    result_dict=data.to_dict()

    return result_dict

data=pd.read_csv('label.csv')
df=csv_to_dict(data)
print(df)