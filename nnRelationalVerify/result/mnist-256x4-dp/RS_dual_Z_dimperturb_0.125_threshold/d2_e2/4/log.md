## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00026658


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9881713, 0.9888232, 0.9881713, 0.9888232, -0.0004505, 0.0004505)
1: (-0.0042114, -0.0040489, -0.0042114, -0.0040489, -0.0001122, 0.0001122)
2: (0.0114031, 0.0122641, 0.0114031, 0.0122641, -0.0005949, 0.0005949)
3: (-0.0068552, -0.0064633, -0.0068552, -0.0064633, -0.0002708, 0.0002708)
4: (0.0027349, 0.0029016, 0.0027349, 0.0029016, -0.0001151, 0.0001151)
5: (0.0133016, 0.0143844, 0.0133016, 0.0143844, -0.0007482, 0.0007482)
6: (-0.0021101, -0.0018353, -0.0021101, -0.0018353, -0.0001899, 0.0001899)
7: (-0.0085971, -0.0078860, -0.0085971, -0.0078860, -0.0004913, 0.0004913)
8: (-0.0040853, -0.0037113, -0.0040853, -0.0037113, -0.0002584, 0.0002584)
9: (0.0024396, 0.0028732, 0.0024396, 0.0028732, -0.0002996, 0.0002996)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.28 = 2.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0003248, upper bound: 0.0003248

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002096, upper bound: 0.0002096
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002096, upper bound: 0.0002096
time: 0.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.97 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.97
Output dim: 0, lower bound: -0.0002096, upper bound: 0.0002096
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.97
Output dim: 0, lower bound: -0.0002096, upper bound: 0.0002096

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.61 + 0.97 = 3.57 seconds
