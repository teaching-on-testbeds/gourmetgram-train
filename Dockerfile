FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    prefect \
    mlflow==3.9.0 \
    pytest

# Copy application files
COPY flow.py /app/flow.py
COPY tests/ /app/tests/

# Generate oversized model during build (>200Mi)
COPY generate_oversized_model.py /app/generate_oversized_model.py
RUN python /app/generate_oversized_model.py

# Run training pipeline
ENTRYPOINT ["python", "flow.py"]
