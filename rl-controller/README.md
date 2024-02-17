# RL-based Multi-dimensional Autoscaling

## Create MPA Objects for Application Deployment

Step 1. Create MPAs for the deployed application

See examples in `multidimensional-pod-autoscaler/examples`:

```
# change the MPA name and the namespace in the YAML file if needed
kubectl create -f multidimensional-pod-autoscaler/examples/hamster.yaml
kubectl create -f multidimensional-pod-autoscaler/examples/hamster-mpa.yaml
```

Step 2. Make Sure the Prometheus environment are set

```
export PROM_SECRET=`oc get secrets -n <monitoring-ns> |grep prometheus-k8s-token |head -n 1|awk '{print $1}'`
export PROM_TOKEN=`echo $(oc get secret $PROM_SECRET -n <monitoring-ns> -o json | jq -r '.data.token') | base64 -d`
export PROM_HOST=`oc get route prometheus-k8s -n <monitoring-ns> -o json | jq -r '.spec.host'`
```

Note that the application metrics (e.g., latency) are exported to Promethues for supporting SLO-driven autoscaling.

## Run RL Controller

Step 1. Install packages

```
cd rl-controller
pip3 install -r requirements.txt
```

Step 2. Start RL policy-training stage

```
# training from scratch
python main.py --app_name <app_name> --app_namespace <app_ns> --mpa_name '<mpa_name> --mpa_namespace <mpa_ns>

# using a checkpoint (bootstrapping)
# --use_checkpoint --checkpoint './checkpoints/ppo-ep-xxx.pth.tar'
```

The logs (for metrics and trajectories) will be stored in `./rl-controller/logs` and model checkpoints will be saved at `./rl-controller/checkpoints`.

Step 3. Start RL policy-serving (inference) stage

```
python main.py --inference --use_checkpoint --checkpoint './checkpoints/ppo-ep-xxx.pth.tar'
```

The agent will detect if retraining is needed whenever necessary based on specified reward thresholds.
