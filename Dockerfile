FROM python:3.10.6-buster

COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY depth_planes depth_planes
COPY setup.py setup.py
RUN pip install .

COPY params.py params.py

CMD uvicorn depth_planes.api.fast:app --host 0.0.0.0 --port $PORT
