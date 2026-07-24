## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000623675


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0018407, -0.0011005, -0.0018407, -0.0011005, -0.0003792, 0.0003792)
1: (-0.0089817, -0.0071033, -0.0089817, -0.0071033, -0.0009623, 0.0009623)
2: (0.0294577, 0.0306231, 0.0294577, 0.0306231, -0.0005970, 0.0005970)
3: (0.0023814, 0.0045575, 0.0023814, 0.0045575, -0.0011148, 0.0011148)
4: (-0.0080290, -0.0061183, -0.0080290, -0.0061183, -0.0009789, 0.0009789)
5: (0.0106970, 0.0114207, 0.0106970, 0.0114207, -0.0003708, 0.0003708)
6: (0.0034104, 0.0061722, 0.0034104, 0.0061722, -0.0014148, 0.0014148)
7: (0.9804457, 0.9823782, 0.9804457, 0.9823782, -0.0009900, 0.0009900)
8: (-0.0075295, -0.0054575, -0.0075295, -0.0054575, -0.0010615, 0.0010615)
9: (-0.0013946, -0.0000259, -0.0013946, -0.0000259, -0.0007012, 0.0007012)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 1.57 = 2.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0006350, upper bound: 0.0006350

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006126, upper bound: 0.0006140
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006140, upper bound: 0.0006125
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.67 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.67
Output dim: 7, lower bound: -0.0006126, upper bound: 0.0006140
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.67
Output dim: 7, lower bound: -0.0006140, upper bound: 0.0006125

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.95 + 1.67 = 4.61 seconds
