FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app
RUN pip install pipenv
COPY ["Pipfile","Pipfile.lock","./"]
RUN pipenv install --system --deploy
COPY ["mean_prediction.py", "./"]
EXPOSE 9696
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696","mean_prediction:app"]