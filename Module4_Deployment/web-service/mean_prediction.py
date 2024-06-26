
import pickle
import pandas as pd
import numpy as np
import sys
from flask import Flask,request, jsonify

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def run_model(data, dv, model):
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{data['year']:04d}-{data['month']:02d}.parquet"
    # output_file = f"predictions_output_{year:04d}-{month:02d}.parquet"
    df = read_data(input_file)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    # print(f"Mean: {np.mean(y_pred)}")
    # df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    # pred_series = pd.Series(y_pred,name='duration')
    # df_results = pd.concat([df[['ride_id']],pred_series],axis=1)
    # df_results.to_parquet(
    # output_file,
    # engine='pyarrow',
    # compression=None,
    # index=False)
    return np.mean(y_pred)

app = Flask("mean-duration-prediction")

@app.route('/predict', methods=['POST'])
def run_model_endpoint():
    data = request.get_json()
    mean = run_model(data, dv, model)
    result = {"mean": mean}
    return jsonify(result)  


if(__name__=='__main__'):
    print("Running the app")
    app.run(debug=True, host='0.0.0.0',port=9696)