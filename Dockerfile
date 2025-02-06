FROM python:3.12.8

# Set working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY .env ./

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
