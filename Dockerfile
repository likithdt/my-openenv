FROM python:3.10-slim

WORKDIR /app

ENV CACHE_REBOOT=6.0

# 1. Copy the requirements file first
COPY requirements.txt .

# 2. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of your code
COPY . .

# 4. Set the path and start command
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["python", "-m", "server.app"]