FROM python:3.12-slim
RUN pip install --no-cache-dir prometheus-client nvidia-ml-py redis docker "psycopg[binary]"
WORKDIR /app
COPY exporter.py events.py ./
ENV PORT=9200
EXPOSE 9200
CMD ["python", "-u", "exporter.py"]
