FROM docker.io/library/python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir dx-com==2.4.0 pillow
