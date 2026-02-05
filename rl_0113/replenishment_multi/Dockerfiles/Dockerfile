FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /workspace

ENV PROMETHEUS_MULTIPROC_DIR /workspace/metrics
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_CONFIG_FILE=/dev/null

RUN mkdir -p /workspace/metrics
RUN python3.8 -m pip install --no-cache-dir --upgrade pip
RUN python3.8 -m pip install --no-cache-dir --trusted-host pypi.shopee.io -i http://pypi.shopee.io/simple/ aip-infer-utils
RUN python3.8 -m pip install --no-cache-dir --trusted-host pypi.shopee.io -i http://pypi.shopee.io/simple/ aip-oneservice
RUN python3.8 -m pip install onnxruntime

COPY conf /workspace/conf
COPY server /workspace/
COPY server/server.py /workspace/

# CMD ["oneservice", "--config=./config/config.yaml", "server:app"]