FROM mambaorg/micromamba:latest

USER root
RUN apt update && \
    apt upgrade -y && \
    apt install -y tar gzip libz-dev software-properties-common python3-software-properties automake nano cmake zip wget gcc git build-essential curl gosu libbz2-dev zlib1g-dev gawk libxml2-dev

USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes