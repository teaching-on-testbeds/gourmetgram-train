FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    mlflow==3.9.0 \
    pytest

# Copy application files
COPY flow.py /app/flow.py
COPY food11.pth /app/food11.pth
COPY tests/ /app/tests/

# Run training pipeline
ENTRYPOINT ["python", "flow.py"]
