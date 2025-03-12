FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye AS build

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpq-dev \
    libopencv-dev \
    libpng-dev \
    libtiff-dev \
    libdcmtk-dev \
    pkg-config \
    libatlas-base-dev \
    libturbojpeg-dev \
    libxml2-dev \
    libxslt1-dev \
    libssl-dev \
    libffi-dev \
    curl \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/

RUN pip3 install --no-cache-dir --only-binary=:all: -r /workspace/requirements.txt

# Final image for running the application
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

WORKDIR /workspace

COPY --from=build /workspace/src /workspace/src
COPY --from=build /workspace/requirements.txt /workspace/requirements.txt

EXPOSE 8000

CMD ["bash"]
