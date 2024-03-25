FROM python:3.8
ENV SRC_DIR /usr/bin/src/webapp/src
COPY web-server-base64.py ${SRC_DIR}/
RUN apt update
RUN apt install -y curl imagemagick libgnutls30
WORKDIR ${SRC_DIR}
ENV PYTHONUNBUFFERED=1
CMD ["python", "web-server-base64.py"]
