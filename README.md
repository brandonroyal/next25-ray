# Ray Next'25 Spotlight Demo

## TODO
* Move to demo project
* Update to "inference" namespace

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

5. Open Notebook and show how easy it is for users to do ray

Run a Job
```bash
kubectl ray job submit --name fine-tune --worker-replicas 3 --worker-cpu 1 --worker-gpu 1 \
  --runtime-env ./runtimeEnv.yaml \
  --working-dir . -- python fine-tune.py
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

2. Install KubeRay patch (temporary)

```bash
git clone https://github.com/ray-project/kuberay
cd kuberay/kubectl-plugin/
```

```bash
go build cmd/kubectl-ray.go
mv kubectl-ray ~/.krew/bin
```

Check that kubectl ray plugin is 'development'. Ignore KubeRay install cannot be found warning.
```bash
kubectl ray version
```

2. Install Ray

```bash
$ sudo pip install ray[default]
```

<!-- 3. Install Kubernetes Python client

```bash
$ pip install kubernetes
``` -->

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

Build Customer Fine-Tune Image (Optional)
```bash
REGION="us-east4"
REPO_NAME="next-demo"
NODE_TRAIN_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/fine-tune:v1"
```

```bash
gcloud artifacts repositories create next-demo --repository-format=docker \
    --location=${REGION}
```

```bash
gcloud builds submit --region=${REGION} --tag=${NODE_TRAIN_IMAGE} --machine-type=${BUILD_MACHINE_TYPE} --timeout=3600 .
```

### Appendix
```bash
kubectl ray job submit --name rayjob-sample --working-dir ./ -- python my_script.py
```

```bash
kubectl ray job submit --dry-run --name rayjob-sample --ray-version 2.41.0 --image rayproject/ray:2.41.0 --head-cpu 1 --head-memory 5Gi --worker-replicas 3 --worker-cpu 1 --worker-memory 5Gi --runtime-env ./runtimeEnv.yaml  -- python my_script.py
```

### CCC FlexStart

export CLOUDSDK_API_ENDPOINT_OVERRIDES_CONTAINER=https://test-container.sandbox.googleapis.com/


export PROJECT_NAME=broyal-serviceproject1
export CLUSTER_NAME=flex-start-demo-cluster
export LOCATION=us-central1-a
export MACHINE_TYPE=g2-standard-8
export ACCELERATOR_COUNT=1
export ACCELERATOR_TYPE=nvidia-l4
export COMPUTE_CLASS_NAME=gpu-class

gcloud container clusters create ${CLUSTER_NAME} \
  --zone ${LOCATION} \
  --project ${PROJECT_NAME} \
  --cluster-version 1.32.2-gke.1475000

gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --zone ${LOCATION} \
  --project ${PROJECT_NAME}

gcloud container node-pools create on-demand-g2-standard-8 \
  --accelerator type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT},gpu-driver-version=LATEST \
  --machine-type ${MACHINE_TYPE} \
  --region ${LOCATION} \
  --cluster ${CLUSTER_NAME} \
  --node-locations ${LOCATION} \
  --node-labels="cloud.google.com/compute-class=${COMPUTE_CLASS_NAME}" \
  --node-taints="cloud.google.com/compute-class=${COMPUTE_CLASS_NAME}:NoSchedule" \
  --num-nodes 0 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 8

ENDPOINT="https://test-container.sandbox.googleapis.com"
METHOD="v1/projects/${PROJECT_NAME}/locations/${LOCATION}/clusters/${CLUSTER_NAME}/nodePools"

curl "${ENDPOINT}/${METHOD}" --request POST --header "Content-type: application/json" --header "Authorization: Bearer $(gcloud auth print-access-token)" --data '
{
  "nodePool": {
    "name": "flex-start-g2-standard-8",
    "locations": ["'${LOCATION}'"],
    "config": {
      "accelerators": [{
        "acceleratorCount": "'${ACCELERATOR_COUNT}'",
        "acceleratorType": "'${ACCELERATOR_TYPE}'",
        "gpuDriverInstallationConfig": {
          "gpuDriverVersion": "LATEST"
        }
      }],
      "reservationAffinity": {
        "consumeReservationType": "NO_RESERVATION"
      },
      "machineType": "'${MACHINE_TYPE}'",
      "diskSizeGb": 100,
      "diskType": "pd-balanced",
      "flexStart": true,
      "maxRunDuration": "86400s" 
    },                                                         
    "initialNodeCount": 0,                              
    "autoscaling": {
      "enabled": true,
      "locationPolicy": "ANY",
      "maxNodeCount": 100
    },                           
    "management": {                                            
      "autoUpgrade": true,                        
      "autoRepair": false                     
    }
  }
}'

gcloud container node-pools update flex-start-g2-standard-8 \
  --cluster ${CLUSTER_NAME} \
  --zone ${LOCATION} \
  --project ${PROJECT_NAME} \
  --node-labels="cloud.google.com/compute-class=gpu-class"

gcloud container node-pools update flex-start-g2-standard-8 \
  --cluster ${CLUSTER_NAME} \
  --zone ${LOCATION} \
  --project ${PROJECT_NAME} \
  --node-taints="cloud.google.com/compute-class=gpu-class:NoSchedule"



### Friction Log
