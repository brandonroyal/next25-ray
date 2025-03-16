# Ray Next'25 Spotlight Demo

## Demo
1. Show variety of workloads on existing cluster `demo-cluster` in Pantheon - existing deployments, jobs, etc.

2. Update existing cluster to support Ray workloads in Pantheon

3. While waiting for cluster to update - switch to `demo-cluster-1`

4. Enable ML Engineers to submit Ray Jobs - deploy a Ray Cluster from the CLI with two commands

```bash
kubectl ray create cluster ml --worker-replicas 3 --worker-cpu 1 --worker-gpu 1
```

```bash
kubectl ray session raycluster/ml
```

5. Open Notebook and show how easy it is for users

Run a Job
```bash
kubectl ray job submit --name fine-tune --working-dir . -- python gemma-fine-tune.py
```
5. While fine-tuning is running, we 



## Client Setup

1. Install kubectl ray plugin

```bash
(
  set -x; cd "$(mktemp -d)" &&
  OS="$(uname | tr '[:upper:]' '[:lower:]')" &&
  ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/\(arm\)\(64\)\?.*/\1\2/' -e 's/aarch64$/arm64/')" &&
  KREW="krew-${OS}_${ARCH}" &&
  curl -fsSLO "https://github.com/kubernetes-sigs/krew/releases/latest/download/${KREW}.tar.gz" &&
  tar zxvf "${KREW}.tar.gz" &&
  ./"${KREW}" install krew
)
```

Add to end of ~/.bashrc file.  Save and re-open terminal
```bash
export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"
```

```bash
$ kubectl krew update
$ kubectl krew install ray
```

2. Install Ray

```bash
$ pip install ray[default]
```

3. Install Kubernetes Python client

```bash
$ pip install kubernetes
```

2. Clone the repository

```bash
$ git clone <repo>
```

## Instructure Setup (Skip if using a shared demo project)

1. Install Cluster

Set variables
```bash
export PROJECT_ID=$(gcloud config get project)
export REGION=us-east4
export ZONE=${REGION}-b
export CLUSTER_NAME=demo-cluster
```

Create GKE Cluster
```bash
gcloud container clusters create $CLUSTER_NAME \
  --location $ZONE \
  --addons=RayOperator \
  --enable-ray-cluster-logging \
  --enable-ray-cluster-monitoring
```

Create a GPU node pool
```bash
## TODO
```

### Appendix
```bash
kubectl ray job submit --name rayjob-sample --working-dir ./ -- python my_script.py
```

```bash
kubectl ray job submit --dry-run --name rayjob-sample --ray-version 2.41.0 --image rayproject/ray:2.41.0 --head-cpu 1 --head-memory 5Gi --worker-replicas 3 --worker-cpu 1 --worker-memory 5Gi --runtime-env ./runtimeEnv.yaml  -- python my_script.py