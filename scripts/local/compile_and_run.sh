#!/bin/bash

if [ -z "${ZONE+x}" ]; then
    echo "Error: ZONE is not set, is your .env loaded?" >&2
    exit 1
fi

BUNDLE_NAME="repo-$(date +%Y%m%d-%H%M%S).bundle"
BASE_NAME="${BUNDLE_NAME%.bundle}"

git bundle create $BUNDLE_NAME --all
gcloud compute scp $BUNDLE_NAME $USER@cuda-gpu:~/$BUNDLE_NAME --zone=$ZONE
rm $BUNDLE_NAME

gcloud compute ssh cuda-gpu --zone=$ZONE --command "$(cat << EOF
set -e

git clone $BUNDLE_NAME
rm $BUNDLE_NAME

EOF
)"

gcloud compute ssh --zone=$ZONE cuda-gpu \
    --command "bash -lc 'cd \"$BASE_NAME\" && ./scripts/remote/compile_and_run.sh'"
