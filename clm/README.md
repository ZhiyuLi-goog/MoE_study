# Mixtral 8x7B with NeMo on GPU Device

## Training

### Docker Image

Build and push docker image:

```shell
docker build -t <regristry_path_image_name>:<image_tag> -f nemo_example.Dockerfile .
docker push <registry_path_image_name>:<image_tag>
```

### Run workflow

In order for this workflow to function, in the ```helm-context``` directory, there must exist a **_select-configuration.yaml_** file.

Package and schedule job. An example job name could be "nemo-gpt3-175b-nemo-16gpus". Use whatever is convenient when searching for later.


```shell
helm install <username_workload_job_name> helm-context/
```

### Monitor workflow

Check pod status (use this to find the name of the pod you want logs from)


```shell
kubectl get pods | grep "<some_part_of_username_workload_job_name>"
```


Check job status


```shell
kubectl get jobs | grep "<some_part_of_username_workload_job_name>"
```


Get logs (Using pod name from earlier)


```shell
kubectl logs "<pod_name>"
```