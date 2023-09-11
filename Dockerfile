####### 👇 OPTIMIZED SOLUTION (x86)👇 #######

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
#FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen
FROM python:3.10.12-slim-bullseye

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY stroke stroke
COPY setup.py setup.py
RUN pip install .

CMD uvicorn stroke.API.fast:app --host 0.0.0.0 --port $PORT

#incorporate somehow: /Users/jmadu1/Documents/healthcare-dataset-stroke-data.csv

#http://0.0.0.0:8000/docs#/default/predict_predict_get
