#!/bin/bash

gcloud compute ssh cuda-gpu --zone=$ZONE --command "$(cat << EOF
set -e

rm -rf *

EOF
)"

