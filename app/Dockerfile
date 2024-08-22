FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV PORT=8080

# Use exec or sh to correctly expand $PORT
CMD exec gunicorn -b :${PORT} app:app
