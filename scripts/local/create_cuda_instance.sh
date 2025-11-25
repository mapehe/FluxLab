#!/bin/bash

gcloud compute instances create cuda-gpu --project=$PROJECT_ID
  --zone=$ZONE --machine-type=n1-standard-1 \
  --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
  --maintenance-policy=TERMINATE --provisioning-model=STANDARD \
  --service-account=$SERVICE_ACCOUNT \
  --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
  --accelerator=count=1,type=nvidia-tesla-t4 \
  --create-disk=auto-delete=yes,boot=yes,device-name=cuda-gpu,image=projects/ml-images/global/images/c0-deeplearning-common-cu124-v20250325-debian-11-py310-conda,mode=rw,size=50,type=pd-balanced \
  --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring \
  --labels=goog-ec-src=vm_add-gcloud --reservation-affinity=any && gcloud compute \
  resource-policies create snapshot-schedule default-schedule-1 \
  --project=$PROJECT_ID --region=$ZONE --max-retention-days=14 \
  --on-source-disk-delete=keep-auto-snapshots --daily-schedule --start-time=15:00 \
  && gcloud compute disks add-resource-policies cuda-gpu --project=$PROJECT_ID \
  --zone=$ZONE \
  --resource-policies=projects/$PROJECT_ID/regions/$REGION/resourcePolicies/default-schedule-1
