#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[7]:


import pickle
import pandas as pd
import numpy as np


# In[5]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[2]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[3]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[6]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[8]:


print(np.std(y_pred))


# In[11]:


print(type(y_pred))


# In[10]:


year = 2023
month = 3


# In[14]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
pred_series = pd.Series(y_pred,name='duration')
df_results = pd.concat([df[['ride_id']],pred_series],axis=1)


# In[16]:


df_results.to_parquet(
    'predictions.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:




