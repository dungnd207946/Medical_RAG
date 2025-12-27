FROM python:3.11-slim
WORKDIR /app

# Cài MkDocs
RUN pip install --no-cache-dir mkdocs mkdocs-material

# Chỉ copy file cần thiết
COPY mkdocs.yml .
COPY docs ./docs

EXPOSE 8000

CMD ["mkdocs", "serve", "-a", "0.0.0.0:8000"]
