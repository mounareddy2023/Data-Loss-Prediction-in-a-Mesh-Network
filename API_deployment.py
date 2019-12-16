
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score

app = Flask(__name__)
HTTP_BAD_REQUEST = 400

MODEL = joblib.load('Mesh_pkl_file_10k.pkl')
MODEL_LABELS = ['Data Loss', 'Data Stable','Data Unstable']



@app.route('/predict1',methods=['POST'])
def predict():
    try:
        #capture json request
        response = request.get_json(force=True)
        #convert to dataframe
        data_raw = pd.DataFrame.from_dict(response, orient='columns')   
        data_target = pd.DataFrame(data_raw[['target']])
        
        data_raw['rssi'] = pd.DataFrame(data_raw['rssi'].astype('int64'))
        data_raw['sequence'] = pd.DataFrame(data_raw['sequence'].astype('int64'))
        data_raw['battery_vdd'] = pd.DataFrame(data_raw['battery_vdd'].astype('float64'))
        data_raw['battery_temp'] = pd.DataFrame(data_raw['battery_temp'].astype('float64'))
        #data_raw['interval'] = pd.DataFrame(data_raw['interval'].astype('float64'))
        
        data_raw=data_raw[['rssi', 'sequence', 'battery_vdd', 'battery_temp']]
    
        print(data_raw)
    # Use the model to predict the class
        label_index = MODEL.predict(data_raw)
        
    # Create and send a response to the API caller
        label_index = pd.DataFrame(label_index)
        
        print("accuracy_score: ",accuracy_score(data_target.target,label_index))
        
        response = jsonify(status='complete', label=label_index.to_json())
        return response
    except Exception as err:
        
        message = ('Failed Exception: {}'.format(err))
        response = jsonify(status='error', error_message=message)
        response.status_code = HTTP_BAD_REQUEST
        return response
        
        
if __name__ == '__main__':
    #set it to False if running in jupyter 
    #set it to True if running in command promt(for debugging)
    app.run(debug=False)