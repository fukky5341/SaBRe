## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00014742


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0061135, 0.0063330, 0.0061135, 0.0063330, -0.0001080, 0.0001080)
1: (-0.0000766, 0.0003485, -0.0000766, 0.0003485, -0.0002091, 0.0002091)
2: (0.0141173, 0.0175461, 0.0141173, 0.0175461, -0.0016869, 0.0016869)
3: (-0.0040797, -0.0037735, -0.0040797, -0.0037735, -0.0001507, 0.0001507)
4: (0.0015422, 0.0030280, 0.0015422, 0.0030280, -0.0007310, 0.0007310)
5: (-0.0009718, -0.0007500, -0.0009718, -0.0007500, -0.0001091, 0.0001091)
6: (0.9916286, 0.9920353, 0.9916286, 0.9920353, -0.0002001, 0.0002001)
7: (-0.0105911, -0.0079016, -0.0105911, -0.0079016, -0.0013232, 0.0013232)
8: (-0.0023298, -0.0014872, -0.0023298, -0.0014872, -0.0004146, 0.0004146)
9: (-0.0043610, -0.0026792, -0.0043610, -0.0026792, -0.0008274, 0.0008274)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.55 + 1.43 = 2.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0001492, upper bound: 0.0001492

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001327, upper bound: 0.0001326
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001327, upper bound: 0.0001326
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.10 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.10
Output dim: 6, lower bound: -0.0001327, upper bound: 0.0001326
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.10
Output dim: 6, lower bound: -0.0001327, upper bound: 0.0001326

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.97 + 1.10 = 4.07 seconds
