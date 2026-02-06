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
COPY food11.pth /app/food11.pth
COPY oversized_model.py /app/oversized_model.py
COPY create_failure_models.py /app/create_failure_models.py
COPY tests/ /app/tests/

# Generate failure scenario models during build
RUN python /app/create_failure_models.py

# Run training pipeline
ENTRYPOINT ["python", "flow.py"]
