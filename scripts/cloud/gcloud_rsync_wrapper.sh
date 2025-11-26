HOST="$1"

shift

exec gcloud compute ssh "$HOST" --zone="$ZONE" --project="$PROJECT_ID" --quiet --ssh-flag="-T" -- "$@"
