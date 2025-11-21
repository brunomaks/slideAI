FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /workspace

RUN apt-get update && apt-get install -y git vim && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/train.py"]
