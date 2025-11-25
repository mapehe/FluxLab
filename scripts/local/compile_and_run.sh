#!/bin/bash

BUNDLE_NAME="repo-$(date +%Y%m%d-%H%M%S).bundle"
BASE_NAME="${BUNDLE_NAME%.bundle}"

git bundle create $BUNDLE_NAME --all
gcloud compute scp $BUNDLE_NAME $USER@cuda-gpu:~/$BUNDLE_NAME
rm $BUNDLE_NAME

gcloud compute ssh cuda-gpu --zone=$ZONE --command "$(cat << EOF
set -e

git clone $BUNDLE_NAME

EOF
)"

gcloud compute ssh cuda-gpu \
    --command "bash -lc 'cd \"$BASE_NAME\" && ./scripts/remote/compile_and_run.sh'"
