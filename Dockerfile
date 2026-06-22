FROM jupyter/pyspark-notebook:latest

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl jq postgresql-client docker.io && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

RUN pip install --no-cache-dir requests pandas matplotlib seaborn kafka-python prometheus-client
