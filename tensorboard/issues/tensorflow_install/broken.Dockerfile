#This is a sample Image 
FROM ubuntu:latest

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libhdf5-dev \
    python3-h5py \
    wget \
    build-essential

RUN wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.4.3/hdf5-1.14.4-3.tar.gz
RUN tar -xf hdf5-1.14.4-3.tar.gz
RUN cd hdf5-1.14.4-3
RUN ./configure
RUN make -j9
RUN make install
ENV HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial"

# RUN apt-get clean && rm -rf /var/lib/apt/lists/*do

# RUN mkdir /train
# RUN mkdir /train/logs
# RUN mkdir /train/logs/fit
# COPY test/test.py /train/
# COPY test/requirements.txt /train/
COPY ./test/logs /train/logs
COPY --chmod=744 ./test/startup.sh /train/startup.sh
WORKDIR /train

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN python -m pip install --upgrade pip
RUN pip install tensorflow
RUN pip install tensorboard
ENTRYPOINT ["sh", "/train/startup.sh"]
# RUN python ./test.py
