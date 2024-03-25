# AWARE

AWARE is an extensible framework for deploying and managing RL (reinforcement learning)-based agents in production systems.
AWARE provides (1) fast adaptation with meta-learning, (2) reliable RL exploration with bootstrapping, (3) timely retraining by continuous monitoring.

## Requirements

Test setup (Ubuntu version >=18.04 and Docker version >=19.03 should work):
- Ubuntu 18.04
- Docker version 19.03.13
- Python3 (tested on Python 3.10)

```
git clone https://gitlab.engr.illinois.edu/DEPEND/aware.git

cd aware/rl-controller
pip install -r requirements.txt
```

## Directory Structures

```shell
aware/
├── baselines  # RL algorithm implementations in FIRM (DDPG), DQN, and PG
│   ├── dqn
│   ├── firm
│   └── pg
├── benchmarks  # examples of serverless benchmarks
├── deploy-mpa  # scripts to deploy MPA (multi-dimensional pod autoscaler) in Kubernetes
├── multidimensional-pod-autoscaler  # source code of MPA
├── rl-controller  # implementation of RL-based controller
│   ├── blocks.py
│   ├── main.py
│   ├── meta_ppo.py  # meta-learning-assisted RL
│   ├── ppo.py       # non-meta-learning-based RL
│   ├── prometheus_adaptor.py
│   ├── requirements.txt
│   ├── rl_env.py    # RL environment wrapper
│   ├── rnn.py       # RNN-based embedding generation
│   ├── roundTime.py
│   └── util.py      # helper functions (e.g., reward function)
├── synthetic-app-generator  # source code of synthetic application generator
│   ├── function_generator
│   └── function_segments    # example function segments (adopted from Sizeless)
└── testing  # testing with collected dataset
```

## Testing

Start training with meta-learner, with model checkpoints saved to `testing/checkpoints/`.

```
$ cd testing
$ python3 main.py
********** Iteration 0 ************
Memory use (MB): 178.5 CPU util (%): 0.0
Iteration: 0 - Average rewards across episodes: -132.4 | Moving average: -132.4
********** Iteration 1 ************
Memory use (MB): 275.69140625 CPU util (%): 49.575
Iteration: 1 - Average rewards across episodes: -86.0 | Moving average: -109.2
********** Iteration 2 ************
Memory use (MB): 285.60546875 CPU util (%): 49.75
Iteration: 2 - Average rewards across episodes: -12.996 | Moving average: -77.132
********** Iteration 3 ************
Memory use (MB): 305.4375 CPU util (%): 49.76875
Iteration: 3 - Average rewards across episodes: 18.876 | Moving average: -53.13
********** Iteration 4 ************
Memory use (MB): 310.9140625 CPU util (%): 49.46875
Iteration: 4 - Average rewards across episodes: 34.184 | Moving average: -35.667

... ...
```

The final results will be stored in `testing/ppo/`.
The number of episodes is set to 500 for demonstration purpose.

## Deployment of MPA for RL

For running AWARE (training and policy-serving) with application `deployment` managed by RL-based MPA in Kubernetes cluster, follow `multidimensional-pod-autoscaler/README.md` to deploy MPA and `rl-controller/README.md` for training and customizing for application deployments from scratch.

## Reference

```
@inproceedings {qiu2023aware,
  author = {Qiu, Haoran and Mao, Weichao and Wang, Chen and Franke, Hubertus and Kalbarczyk, Zbigniew T. and Ba\c{s}ar, Tamer and Iyer, Ravishankar K.},
  title = {{AWARE}: Automate Workload Autoscaling with Reinforcement Learning in Production Cloud Systems},
  booktitle = {2023 USENIX Annual Technical Conference (USENIX ATC 23)},
  year = {2023},
  address = {Boston, MA},
  pages = {1--17},
  publisher = {USENIX Association},
  month = jul,
}
```

## Contact

Haoran Qiu (haoranq4@illinois.edu)
