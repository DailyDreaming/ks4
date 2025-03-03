#!/bin/bash

YAML_FILE="ks4stitch.yaml"

export DATASET_PATHS="s3://braingeneersdev/asrobbin/24-01-07_data/20217/exp1/exp1.raw.h5, s3://braingeneersdev/asrobbin/24-01-07_data/20217/exp1/exp1_causal.raw.h5"
echo "Processing dataset: $DATASET_PATHS"

export UPLOAD_PATHS="s3://braingeneersdev/test/"
echo "Upload Path: $UPLOAD_PATHS"

export START_DATE="$(date +%s)"
export NAME="ks4-${START_DATE}"
export TAG="$(openssl rand -hex 3)"

# Build and push the Docker image
docker build -f Dockerfile -t quay.io/ucsc_cgl/ks4:${TAG} .
docker push quay.io/ucsc_cgl/ks4:${TAG}

# Substitute the environment variables
yaml=$(envsubst < "$YAML_FILE")

# Apply the modified YAML using kubectl
echo "$yaml" | kubectl apply -f -
