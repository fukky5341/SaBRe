## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00010188


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040959, -0.0040885, -0.0040959, -0.0040885, -0.0000035, 0.0000035)
1: (-0.0060486, -0.0057717, -0.0060486, -0.0057717, -0.0001295, 0.0001295)
2: (0.9692050, 0.9695371, 0.9692050, 0.9695371, -0.0001554, 0.0001554)
3: (0.0191668, 0.0216171, 0.0191668, 0.0216171, -0.0011464, 0.0011464)
4: (-0.0023371, -0.0021508, -0.0023371, -0.0021508, -0.0000872, 0.0000872)
5: (0.0149083, 0.0150966, 0.0149083, 0.0150966, -0.0000881, 0.0000881)
6: (0.0045638, 0.0046554, 0.0045638, 0.0046554, -0.0000429, 0.0000429)
7: (-0.0133805, -0.0127455, -0.0133805, -0.0127455, -0.0002971, 0.0002971)
8: (0.0061137, 0.0066175, 0.0061137, 0.0066175, -0.0002357, 0.0002357)
9: (0.0087206, 0.0096268, 0.0087206, 0.0096268, -0.0004239, 0.0004239)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.25 = 2.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0001026, upper bound: 0.0001026

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0000977, upper bound: 0.0000989
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0000989, upper bound: 0.0000977
time: 0.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.99 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.99
Output dim: 2, lower bound: -0.0000977, upper bound: 0.0000989
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.99
Output dim: 2, lower bound: -0.0000989, upper bound: 0.0000977

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.75 + 0.99 = 3.74 seconds
