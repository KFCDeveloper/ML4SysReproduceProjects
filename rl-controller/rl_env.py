import os
import time
from util import *
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from prometheus_adaptor import *


MPA_DOMAIN = 'autoscaling.k8s.io'
MPA_VERSION = 'v1alpha1'
MPA_PLURAL = 'multidimpodautoscalers'

# system metrics are exported to system-space Prometheus instance
PROM_URL = 'http://localhost:9090'
PROM_TOKEN = None  # if required, to be overwritten by the environment variable

# custom metrics are exported to user-space Prometheus instance
PROM_URL_USERSPACE = 'http://localhost:9090'
# PROM_URL_USERSPACE = 'https://prometheus-k8s.openshift-monitoring.svc.cluster.local:9091'  # used when running the RL controller as a pod in the cluster
PROM_TOKEN_USERSPACE = None  # user-space Prometheus does not require token access

FORECASTING_SIGHT_SEC = 30        # look back for 30s for Prometheus data
HORIZONTAL_SCALING_INTERVAL = 10  # wait for 10s for horizontal scaling to update
VERTICAL_SCALING_INTERVAL = 10    # wait for 10s for vertical scaling to update

class RLEnvironment:
    app_name = 'my-app'
    app_namespace = 'my-app-ns'
    mpa_name = 'my-mpa'
    mpa_namespace = 'my-mpa-ns'

    # initial resource allocations
    initial_num_replicas = 1
    initial_cpu_limit = 1024      # millicore
    initial_memory_limit = 2048   # MiB

    # states
    controlled_resources = ['cpu', 'memory', 'blkio']
    # CHANGEME: [Add application-related custom metrics here]
    custom_metrics = [
        'event_file_discovery_rate',
        'event_rate_processing', 'event_rate_ingestion', 'event_rate',
        'event_latency'
    ]
    states = {
        # system-wise metrics
        'cpu_util': 0.0,                   # 0
        'memory_util': 0.0,                # 1
        'disk_io_usage': 0.0,              # 2
        # 'ingress_rate': 0.0,
        # 'egress_rate': 0.0,
        # application-wise metrics
        'file_discovery_rate': 0.0,        # 3
        'rate': 0.0,  # 1 / lag            # 4
        'processing_rate': 0.0,            # 5
        'ingestion_rate': 0.0,             # 6
        'latency': 0.0,                    # 7
        # resource allocation
        'num_replicas': 1,                 # 8
        'cpu_limit': 1024,                 # 9
        'memory_limit': 2048,              # 10
    }

    # action and reward at the previous time step
    last_action = {
        'vertical_cpu': 0,
        'vertical_memory': 0,
        'horizontal': 0
    }
    last_reward = 0

    def __init__(self, app_name='my-app', app_namespace='my-app-ns', mpa_name='my-mpa', mpa_namespace='my-mpa-ns'):
        self.app_name = app_name
        self.app_namespace = app_namespace
        self.mpa_name = mpa_name
        self.mpa_namespace = mpa_namespace

        # load cluster config
        if 'KUBERNETES_PORT' in os.environ:
            config.load_incluster_config()
        else:
            config.load_kube_config()

        # get the api instance to interact with the cluster
        api_client = client.api_client.ApiClient()
        self.api_instance = client.AppsV1Api(api_client)
        self.corev1 = client.CoreV1Api(api_client)

        # set up the prometheus client
        global PROM_URL
        if os.getenv("PROM_HOST"):
            user_specified_prom_host = os.getenv("PROM_HOST")
            if not user_specified_prom_host in PROM_URL:
                PROM_URL = 'https://' + user_specified_prom_host
                print('PROM_URL is set to:', PROM_URL)
        if not os.getenv("PROM_TOKEN"):
            print("PROM_TOKEN not set! Please set PROM_URL and PROM_TOKEN properly in the environment variables.")
            exit()
        else:
            PROM_TOKEN = os.getenv("PROM_TOKEN")
        self.prom_client = PromCrawler(prom_address=PROM_URL, prom_token=PROM_TOKEN)
        self.user_space_prom_client = PromCrawler(prom_address=PROM_URL_USERSPACE, prom_token=PROM_TOKEN_USERSPACE)

        # current resource limit
        self.states['cpu_limit'] = self.initial_cpu_limit
        self.states['memory_limit'] = self.initial_memory_limit
        self.states['num_replicas'] = self.initial_num_replicas

        # self.observe_states()
        self.init()

    # observe the current states
    def observe_states(self):
        target_containers = self.get_target_containers()
        target_containers = [container for container in target_containers if not 'log' in container]
        print('target_containers:', target_containers)

        # get system metrics for target containers
        traces = {}
        namespace_query = "namespace=\'" + self.app_namespace + "\'"
        container_queries = []
        self.prom_client.update_period(FORECASTING_SIGHT_SEC)
        self.user_space_prom_client.update_period(FORECASTING_SIGHT_SEC)
        for container in target_containers:
            container_query = "container='" + container + "'"
            container_queries.append(container_query)

        for resource in self.controlled_resources:
            if resource.lower() == "cpu":
                resource_query = "rate(container_cpu_usage_seconds_total{%s}[1m])"
            elif resource.lower() == "memory":
                resource_query = "container_memory_usage_bytes{%s}"
            elif resource.lower() == "blkio":
                resource_query = "container_fs_usage_bytes{%s}"
            elif resource.lower() == "ingress":
                resource_query = "rate(container_network_receive_bytes_total{%s}[1m])"
            elif resource.lower() == "egress":
                resource_query = "rate(container_network_transmit_bytes_total{%s}[1m])"

            # retrieve the metrics for target containers in all pods
            for container_query in container_queries:
                query_index = namespace_query + "," + container_query
                query = resource_query % (query_index)
                print(query)

                # retrieve the metrics for the target container from Prometheus
                traces = self.prom_client.get_promdata(query, traces, resource)

        # custom metrics exported to prometheus
        for metric in self.custom_metrics:
            metric_query = metric.lower()
            print(metric_query)

            # retrieve the metrics from Prometheus
            traces = self.user_space_prom_client.get_promdata(metric_query, traces, metric)

        # print('Collected Traces:', traces)
        print('Collected traces for', self.app_name)
        cpu_traces = traces[self.app_name]['cpu']
        memory_traces = traces[self.app_name]['memory']
        blkio_traces = traces[self.app_name]['blkio']
        # ingress_traces = traces[self.app_name]['ingress']
        # egress_traces = traces[self.app_name]['egress']

        # compute the average utilizations
        if 'cpu' in self.controlled_resources:
            all_values = []
            print('cpu_traces:', cpu_traces)
            for container in cpu_traces:
                cpu_utils = []
                for measurement in cpu_traces[container]:
                    cpu_utils.append(float(measurement[1]))
                print('Avg CPU Util ('+container+'):', np.mean(cpu_utils))
                all_values.append(np.mean(cpu_utils))
            self.states['cpu_util'] = np.mean(all_values)
        if 'memory' in self.controlled_resources:
            all_values = []
            for container in memory_traces:
                memory_usages = []
                for measurement in memory_traces[container]:
                    memory_usages.append(int(measurement[1]) / 1024 / 1024.0)
                print('Avg Memory Usage ('+container+'):', np.mean(memory_usages), 'MiB', '| Limit:', self.states['memory_limit'], 'MiB')
                all_values.append(np.mean(memory_usages))
            self.states['memory_util'] = np.mean(all_values) / self.states['memory_limit']
        if 'blkio' in self.controlled_resources:
            all_values = []
            for container in blkio_traces:
                blkio_usages = []
                for measurement in blkio_traces[container]:
                    blkio_usages.append(int(measurement[1]) / 1024 / 1024.0)
                print('Avg Disk I/O Usage ('+container+'):', np.mean(blkio_usages), 'MiB')
                all_values.append(np.mean(blkio_usages))
            self.states['disk_io_usage'] = np.mean(all_values)
        if 'ingress' in self.controlled_resources:
            all_values = []
            for container in ingress_traces:
                ingress = []
                for measurement in ingress_traces[container]:
                    ingress.append(int(measurement[1]) / 1024.0)
                print('Avg Ingress ('+container+'):', np.mean(ingress), 'KiB/s')
                all_values.append(np.mean(ingress))
            self.states['ingress_rate'] = np.mean(all_values)
        if 'egress' in self.controlled_resources:
            all_values = []
            for container in egress_traces:
                egress = []
                for measurement in egress_traces[container]:
                    egress.append(int(measurement[1]) / 1024.0)
                print('Avg egress ('+container+'):', np.mean(egress), 'KiB/s')
                all_values.append(np.mean(egress))
            self.states['egress_rate'] = np.mean(all_values)

        # get the custom metrics (application-related)
        if 'event_file_discovery_rate' in self.custom_metrics:
            metric_traces = traces[self.app_name]['event_file_discovery_rate']
            rate = []
            for trace in metric_traces:
                values = []
                for measurement in metric_traces[trace]:
                    values.append(float(measurement[1]))
                rate.append(np.mean(values))
            # print('Avg file discovery rate:', np.mean(rate))
            # print('Total file discovery rate:', sum(rate))
            self.states['file_discovery_rate'] = np.mean(rate)
        if 'event_rate_processing' in self.custom_metrics:
            metric_traces = traces[self.app_name]['event_rate_processing']
            rate = []
            for container in metric_traces:
                values = []
                for measurement in metric_traces[container]:
                    values.append(float(measurement[1]))
                print(container, np.mean(values))
                rate.append(np.mean(values))
            # print('Avg processing rate:', np.mean(rate))
            # print('Total processing rate:', sum(rate))
            self.states['processing_rate'] = np.mean(rate)
        if 'event_rate_ingestion' in self.custom_metrics:
            metric_traces = traces[self.app_name]['event_rate_ingestion']
            rate = []
            for container in metric_traces:
                values = []
                for measurement in metric_traces[container]:
                    values.append(float(measurement[1]))
                print(container, np.mean(values))
                rate.append(np.mean(values))
            # print('Avg ingestion rate:', np.mean(rate))
            # print('Total ingestion rate:', sum(rate))
            self.states['ingestion_rate'] = np.mean(rate)
        if 'event_rate' in self.custom_metrics:
            metric_traces = traces[self.app_name]['event_rate']
            rate = []
            for container in metric_traces:
                values = []
                for measurement in metric_traces[container]:
                    values.append(float(measurement[1]))
                print(container, np.mean(values))
                rate.append(np.mean(values))
            # print('Avg rate:', np.mean(rate))
            # print('Total rate:', sum(rate))
            self.states['rate'] = np.mean(rate)
        if 'event_latency' in self.custom_metrics:
            metric_traces = traces[self.app_name]['event_latency']
            latency = []
            for container in metric_traces:
                values = []
                for measurement in metric_traces[container]:
                    values.append(float(measurement[1]))
                print(container, np.mean(values))
                latency.append(np.mean(values))
            # print('Avg latency:', np.mean(latency))
            self.states['latency'] = np.mean(latency)

    # initialize the environment
    def init(self):
        # rescale the number of replicas
        self.api_instance.patch_namespaced_deployment_scale(
            self.app_name,
            self.app_namespace,
            {'spec': {'replicas': self.initial_num_replicas}}
        )
        self.states['num_replicas'] = self.initial_num_replicas
        print('Set the number of replicas to:', self.states['num_replicas'])

        # reset the cpu and memory limit
        self.states['cpu_limit'] = self.initial_cpu_limit
        self.states['memory_limit'] = self.initial_memory_limit
        self.set_vertical_scaling_recommendation(self.states['cpu_limit'], self.states['memory_limit'])
        print('Set the CPU limit to:', self.states['cpu_limit'], 'millicores, memory limit to:', self.states['memory_limit'], 'MiB')

        # get the current state
        self.observe_states()

        return self.states

    # reset the environment for RL training (not initializing the environment)
    # just observe the current state and treat it as the initial state
    def reset(self):
        # get the current state
        self.observe_states()

        return self.states

    # action sanity check
    def sanity_check(self, action):
        if action['horizontal'] != 0:
            if self.states['num_replicas'] + action['horizontal'] < MIN_INSTANCES:
                return False
            if self.states['num_replicas'] + action['horizontal'] > MAX_INSTANCES:
                return False
        elif action['vertical_cpu'] != 0:
            cpu_limit_to_set = self.states['cpu_limit'] + action['vertical_cpu']
            if cpu_limit_to_set > MAX_CPU_LIMIT or cpu_limit_to_set < MIN_INSTANCES:
                return False
            if self.states['num_replicas'] <= 1:
                # num_replicas must >= 2 when vertical scaling needs to evict the pod
                return False
        elif action['vertical_memory'] != 0:
            memory_limit_to_set = self.states['memory_limit'] + action['vertical_memory']
            if memory_limit_to_set > MAX_MEMORY_LIMIT or memory_limit_to_set < MIN_MEMORY_LIMIT:
                return False
            if self.states['num_replicas'] <= 1:
                # num_replicas must >= 2 when vertical scaling needs to evict the pod
                return False
        return True

    # get all target container names
    def get_target_containers(self):
        target_pods = self.corev1.list_namespaced_pod(namespace=self.app_namespace, label_selector="app=" + self.app_name)

        target_containers = []
        for pod in target_pods.items:
            for container in pod.spec.containers:
                if container.name not in target_containers:
                    target_containers.append(container.name)

        return target_containers

    # set the vertical scaling recommendation to MPA
    def set_vertical_scaling_recommendation(self, cpu_limit, memory_limit):
        # update the recommendations
        container_recommendation = {"containerName": "", "lowerBound": {}, "target": {}, "uncappedTarget": {}, "upperBound": {}}
        container_recommendation["lowerBound"]['cpu'] = str(cpu_limit) + 'm'
        container_recommendation["target"]['cpu'] = str(cpu_limit) + 'm'
        container_recommendation["uncappedTarget"]['cpu'] = str(cpu_limit) + 'm'
        container_recommendation["upperBound"]['cpu'] = str(cpu_limit) + 'm'
        container_recommendation["lowerBound"]['memory'] = str(memory_limit) + 'Mi'
        container_recommendation["target"]['memory'] = str(memory_limit) + 'Mi'
        container_recommendation["uncappedTarget"]['memory'] = str(memory_limit) + 'Mi'
        container_recommendation["upperBound"]['memory'] = str(memory_limit) + 'Mi'

        recommendations = []
        containers = self.get_target_containers()
        for container in containers:
            vertical_scaling_recommendation = container_recommendation.copy()
            vertical_scaling_recommendation['containerName'] = container
            recommendations.append(vertical_scaling_recommendation)

        patched_mpa = {"recommendation": {"containerRecommendations": recommendations}, "currentReplicas": self.states['num_replicas'], "desiredReplicas": self.states['num_replicas']}
        body = {"status": patched_mpa}
        mpa_api = client.CustomObjectsApi()

        # Update the MPA object
        # API call doc: https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CustomObjectsApi.md#patch_namespaced_custom_object
        try:
            mpa_updated = mpa_api.patch_namespaced_custom_object(group=MPA_DOMAIN, version=MPA_VERSION, plural=MPA_PLURAL, namespace=self.mpa_namespace, name=self.mpa_name, body=body)
            print("Successfully patched MPA object with the recommendation: %s" % mpa_updated['status']['recommendation']['containerRecommendations'])
        except ApiException as e:
            print("Exception when calling CustomObjectsApi->patch_namespaced_custom_object: %s\n" % e)

    # execute the action after sanity check
    def execute_action(self, action):
        if action['vertical_cpu'] != 0:
            # vertical scaling of cpu limit
            self.states['cpu_limit'] += action['vertical_cpu']
            self.set_vertical_scaling_recommendation(self.states['cpu_limit'], self.states['memory_limit'])
            # sleep for a period of time to wait for update
            time.sleep(VERTICAL_SCALING_INTERVAL)
        elif action['vertical_memory'] != 0:
            # vertical scaling of memory limit
            self.states['memory_limit'] += action['vertical_memory']
            self.set_vertical_scaling_recommendation(self.states['cpu_limit'], self.states['memory_limit'])
            # sleep for a period of time to wait for update
            time.sleep(VERTICAL_SCALING_INTERVAL)
        elif action['horizontal'] != 0:
            # scaling in/out
            num_replicas = self.states['num_replicas'] + action['horizontal']
            self.api_instance.patch_namespaced_deployment_scale(
                self.app_name,
                self.app_namespace,
                {'spec': {'replicas': num_replicas}}
            )
            print('Scaled to', num_replicas, 'replicas')
            self.states['num_replicas'] = num_replicas
            # sleep for a period of time to wait for update
            time.sleep(HORIZONTAL_SCALING_INTERVAL)
        else:
            # no action to perform
            print('No action')
            pass

    # RL step function to update the environment given the input actions
    # action: +/- cpu limit; +/- memory limit; +/- number of replicas
    # return: state, reward
    def step(self, action):
        curr_state = self.states.copy()

        # action correctness check:
        if not self.sanity_check(action):
            self.last_reward = -1
            return curr_state, ILLEGAL_PENALTY

        # execute the action on the cluster
        self.execute_action(action)

        # observe states
        self.observe_states()

        next_state = self.states

        # calculate the reward
        reward = convert_state_action_to_reward(next_state, action, self.last_action, curr_state, app_name=self.app_name)

        self.last_reward = reward
        self.last_action = action

        return next_state, reward

    # print state information
    def print_info(self):
        print('Application name:', self.app_name, '(namespace: ' + self.app_namespace + ')')
        print('Current allocation: CPU limit =', str(self.states['cpu_limit'])+'m', 'Memory limit:', self.states['memory_limit'], 'MiB')
        print('Avg CPU utilization: {:.3f}'.format(self.states['cpu_util']))
        print('Avg memory utilization: {:.3f}'.format(self.states['memory_util']))
        print('Avg disk usage: {:.3f}'.format(self.states['disk_io_usage']))
        print('Avg file discovery rate: {:.3f}'.format(self.states['file_discovery_rate']))
        print('Avg rate: {:.3f}'.format(self.states['rate']))
        print('Avg processing rate: {:.3f}'.format(self.states['processing_rate']))
        print('Avg ingestion rate: {:.3f}'.format(self.states['ingestion_rate']))
        print('Avg latency: {:.3f}'.format(self.states['latency']))
        print('Num of replicas: {:d}'.format(self.states['num_replicas']))


if __name__ == '__main__':
    # testing
    env = RLEnvironment(app_name='hamster', app_namespace='default', mpa_name='hamster-mpa', mpa_namespace='default')
    env.print_info()
