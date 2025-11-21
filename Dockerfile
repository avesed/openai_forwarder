FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY openai-python ./openai-python
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 3000

ENV GUNICORN_TIMEOUT=120
ENV FORWARDER_PORT=3000

CMD ["sh", "-c", "gunicorn --chdir src --bind 0.0.0.0:${FORWARDER_PORT:-3000} --timeout ${GUNICORN_TIMEOUT:-120} server:app"]
