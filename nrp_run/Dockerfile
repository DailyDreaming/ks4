FROM braingeneers/braingeneers:latest

RUN apt-get update && \
    apt-get install -y build-essential wget git time nano glances && \
    rm -rf /var/lib/apt/lists/*

ENV MINICONDA_VERSION 4.8.2
ENV CONDA_DIR /home/miniconda3
ENV LATEST_CONDA_SCRIPT "Miniconda3-py39_$MINICONDA_VERSION-Linux-x86_64.sh"

ENV PATH=$CONDA_DIR/bin:$PATH
RUN pip install git+https://github.com/SpikeInterface/spikeinterface.git
RUN pip install kilosort awscli ipython
ENV CUDA_CACHE_MAXSIZE=8589934592

COPY src/ /src
WORKDIR /src
