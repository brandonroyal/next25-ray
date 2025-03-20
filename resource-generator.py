import random
import string
import yaml

def generate_random_string(length=10):
    """Generates a random string of given length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def generate_random_name(prefix="random"):
    """Generates a random name with a prefix."""
    return f"{prefix}-{generate_random_string(8)}"

def generate_random_deployment(namespace="default"):
    """Generates a random Kubernetes Deployment."""
    name = generate_random_name("deployment")
    replicas = random.randint(1, 5)
    image = f"nginx:{random.randint(1, 2)}" # Simple example, you can expand this
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app": name,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": name,
                    },
                },
                "spec": {
                    "containers": [
                        {
                            "name": "nginx",
                            "image": image,
                        },
                    ],
                },
            },
        },
    }
    return deployment

def generate_random_daemonset(namespace="default"):
    """Generates a random Kubernetes DaemonSet."""
    name = generate_random_name("daemonset")
    image = f"busybox:{random.randint(1, 3)}" # example
    daemonset = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "selector": {
                "matchLabels": {
                    "app": name,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": name,
                    },
                },
                "spec": {
                    "containers": [
                        {
                            "name": "busybox",
                            "image": image,
                            "command": ["/bin/sh", "-c", "while true; do sleep 3600; done"], #example
                        },
                    ],
                },
            },
        },
    }
    return daemonset

def generate_random_job(namespace="default"):
    """Generates a random Kubernetes Job."""
    name = generate_random_name("job")
    image = f"ubuntu:{random.randint(18, 22)}.04" # example
    command = ["/bin/bash", "-c", "echo 'Job executed successfully'"] # example
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "ubuntu",
                            "image": image,
                            "command": command,
                        },
                    ],
                    "restartPolicy": "Never",
                },
            },
            "backoffLimit": 4,
        },
    }
    return job

def generate_kubernetes_resources(num_deployments, num_daemonsets, num_jobs, namespace="default"):
    """Generates a list of random Kubernetes resources."""
    resources = []
    for _ in range(num_deployments):
        resources.append(generate_random_deployment(namespace))
    for _ in range(num_daemonsets):
        resources.append(generate_random_daemonset(namespace))
    for _ in range(num_jobs):
        resources.append(generate_random_job(namespace))
    return resources

def write_yaml_files(resources, output_dir="."):
    """Writes Kubernetes resources to YAML files."""
    for resource in resources:
        filename = f"{output_dir}/{resource['kind'].lower()}-{resource['metadata']['name']}.yaml"
        with open(filename, "w") as f:
            yaml.dump(resource, f)

if __name__ == "__main__":
    num_deployments = 6
    num_daemonsets = 2
    num_jobs = 4
    namespace = "default" # Change as needed
    output_dir = "generated-k8s" # Change as needed.

    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    resources = generate_kubernetes_resources(num_deployments, num_daemonsets, num_jobs, namespace)
    write_yaml_files(resources, output_dir)

    print(f"Generated {num_deployments} Deployments, {num_daemonsets} DaemonSets, and {num_jobs} Jobs in '{output_dir}'.")