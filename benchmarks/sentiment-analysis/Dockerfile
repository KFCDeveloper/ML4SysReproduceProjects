FROM python:3.8
ENV SRC_DIR /usr/bin/src/webapp/src
COPY web-server-senti.py ${SRC_DIR}/
COPY senti-data.json ${SRC_DIR}/
RUN pip install --no-cache-dir -U textblob
RUN python -m textblob.download_corpora
RUN apt update
RUN apt install -y curl imagemagick libgnutls30
WORKDIR ${SRC_DIR}
ENV PYTHONUNBUFFERED=1
CMD ["python", "web-server-senti.py"]
