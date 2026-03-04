FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        "wyoming>=1.5.0" \
        "aiohttp>=3.9"

WORKDIR /app
COPY server.py ./

EXPOSE 10300

ENTRYPOINT ["python3", "/app/server.py"]
