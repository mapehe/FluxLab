#!/bin/bash

if [ -z "${ZONE+x}" ]; then
    echo "Error: ZONE is not set, is your .env loaded?" >&2
    exit 1
fi

cd "$(dirname "$0")/../.."

rsync -avz \
  -e ./scripts/cloud/gcloud_rsync_wrapper.sh \
  --exclude '.git' \
  --exclude 'build_source' \
  --exclude '*.o' \
  ./src ./include Makefile config.json \
  cuda-gpu:~/build_source/

gcloud compute ssh --zone=$ZONE cuda-gpu \
    --command "bash -lc 'cd build_source && make run && gcloud storage cp simulation_result gs://${STORAGE_BUCKET}/simulation_result_\$(date +%Y%m%d_%H%M%S)'"
