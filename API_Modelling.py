import pandas as pd

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 

from sklearn.externals import joblib

data = pd.read_csv('Mesh_train_data.csv')
df = pd.DataFrame(data)

#split our dataset
X = df.loc[:,['rssi', 'sequence','battery_vdd','battery_temp']]
y = df.loc[:,['target']]
y.target.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)


from xgboost import XGBClassifier

XGBmodel = XGBClassifier(learning_rate =0.1,n_estimators=10000,max_depth=9,seed=123)
XGBmodel = XGBmodel.fit(X_train, y_train)

y_predictor = XGBmodel.predict(X_test)
 
results = confusion_matrix(y_test, y_predictor) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(y_test, y_predictor) )
print('Report : ')
print(classification_report(y_test, y_predictor) )

y_test=y_test.reset_index()
y_test=y_test.drop(['index'], axis=1)

y_pred = pd.DataFrame(y_predictor, columns = ['target_pred'])
y_actual = pd.DataFrame(y_test, columns = ['target'])

evaluate = pd.DataFrame() #creates a new dataframe that's empty
evaluate=pd.concat([y_actual,y_pred], axis=1)

print('crosstab :\n',pd.crosstab(evaluate.target,evaluate.target_pred))

print("Score on the training set is: {:2}".format(XGBmodel.score(X_train, y_train)))
print("Score on the test set is: {:.2}".format(XGBmodel.score(X_test, y_test)))

# Save the model
model_filename = 'Mesh_pkl_file_10k.pkl'
print("Saving model to {}".format(model_filename))
joblib.dump(XGBmodel, model_filename)















