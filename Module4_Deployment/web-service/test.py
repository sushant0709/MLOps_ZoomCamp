# from mean_prediction import run_model
import sys
import requests

url = "http://localhost:9696/predict"
if (__name__ =="__main__"):
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    data = {"year": year, "month":month}
    response = requests.post(url,json=data)
    print(response.json())

