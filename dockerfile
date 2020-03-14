FROM pytorch/pytorch:latest

COPY requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt