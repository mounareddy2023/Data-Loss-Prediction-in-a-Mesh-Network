import requests
import pandas as pd
import json
data=pd.read_csv("validation_set1.csv")
data.isnull().values.any()
data.isnull().sum()
data = data.dropna()

data=data[['rssi', 'sequence','battery_vdd','battery_temp','target']]
print("\n •	validation_set1.csv")
print("\n Inputs:\n")
print(data.head(5))
print(data['target'].value_counts())

json_data = data.to_json(orient='records')
url = "http://127.0.0.1:5000/predict1"
r = requests.post(url, data=json_data)
r.text
#print(r.json())
z=r.json()
y = z.get("label")
res = json.loads(y)
data_dict = res.get("0")
data_op = pd.DataFrame.from_dict(data_dict, orient='index')
data_op.columns = ['Mesh_stability']
print("\n Output:\n")
print(data_op.head(5))
print(data_op['Mesh_stability'].value_counts())


pred = pd.DataFrame(data_op.Mesh_stability)
actual = pd.DataFrame(data.target)

evaluate = pd.DataFrame() #creates a new dataframe that's empty
evaluate=pd.concat([actual,pred.reset_index().drop(['index'], axis=1)], axis=1)

print('crosstab :\n',pd.crosstab(evaluate.target,evaluate.Mesh_stability))
