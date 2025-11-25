# Cuda Simulation Experiment

This repo is my personal experiment. I'm interested in running some physics
simulations with CUDA. The idea is to track my progress here.

## How to initialize

Create an `.env` file with the following contents.

```
export USER="your-local-username"
export PROJECT_ID="your-google-cloud-project-name"
export SERVICE_ACCOUNT="your-google-cloud-service-account"
export ZONE="us-east1-d"
export REGION="us-east1"
```

Install `gcloud`, login, make sure you have the permissions to create
GPU instancese.

## How to work

1.  Load the `.env` with `source .env`.
2.  Create a CUDA-enabled instance with `scripts/local/create_cuda_instance.sh`
3.  Connect to the instance with `./scripts/local/ssh_instance.sh` and make
    sure the NVIDIA drivers are installed. (It may take a few minutes for the
    instance to be responsive.)
4.  To compile and run the code on your CUDA instance
    `scripts/local/compile_and_run.sh`
5.  When done, remember to run `scripts/local/delete_cuda_instance.sh` to avoid
    unnecessary bills.
