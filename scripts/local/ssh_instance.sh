#!/bin/bash


if [ -z "${ZONE+x}" ]; then
    echo "Error: ZONE is not set, is your .env loaded?" >&2
    exit 1
fi

gcloud compute ssh cuda-gpu --zone=$ZONE
