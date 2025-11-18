FROM python:3.11 
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "websocket.py"]