FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Solo dependencias mínimas de sistema para ONNX y OpenCV Headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/data

RUN groupadd -r dev -g 1001 \
    && useradd -m -r -g dev -u 1001 -d /home/dev -s /bin/bash dev

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN chown -R dev:dev /app/
USER dev

CMD ["tail", "-f", "/dev/null"]