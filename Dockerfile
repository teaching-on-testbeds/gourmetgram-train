FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY train.py /app/train.py
ENTRYPOINT ["python", "/app/train.py"]
