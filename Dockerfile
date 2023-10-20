FROM tiangolo/uvicorn-gunicorn:python3.9

WORKDIR /app

COPY requirements.txt .
COPY main.py .

# Install dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 80