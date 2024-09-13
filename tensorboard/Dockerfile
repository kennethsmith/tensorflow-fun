#This is a sample Image 
FROM ubuntu:latest

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv

COPY --chmod=744 ./scripts/start-tensorboard.sh /scripts/start-tensorboard.sh

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN python -m pip install --upgrade pip
RUN pip install tensorboard
ENTRYPOINT ["sh", "/scripts/start-tensorboard.sh"]
