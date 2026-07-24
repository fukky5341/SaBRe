## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00071487


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0136107, -0.0095040, -0.0136107, -0.0095040, -0.0029699, 0.0029699)
1: (-0.0067760, -0.0056182, -0.0067760, -0.0056182, -0.0008373, 0.0008373)
2: (-0.0114352, -0.0028924, -0.0114352, -0.0028924, -0.0061781, 0.0061781)
3: (0.0001140, 0.0012445, 0.0001140, 0.0012445, -0.0008176, 0.0008176)
4: (0.0082534, 0.0146378, 0.0082534, 0.0146378, -0.0046171, 0.0046171)
5: (0.9977993, 0.9995731, 0.9977993, 0.9995731, -0.0012828, 0.0012828)
6: (0.0058861, 0.0074961, 0.0058861, 0.0074961, -0.0011644, 0.0011644)
7: (-0.0014157, 0.0045927, -0.0014157, 0.0045927, -0.0043452, 0.0043452)
8: (-0.0127674, -0.0080910, -0.0127674, -0.0080910, -0.0033819, 0.0033819)
9: (-0.0033117, -0.0029082, -0.0033117, -0.0029082, -0.0002918, 0.0002918)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.68 = 3.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0007752, upper bound: 0.0007751

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006933, upper bound: 0.0006933
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006933, upper bound: 0.0006933
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.53 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.53
Output dim: 5, lower bound: -0.0006933, upper bound: 0.0006933
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.53
Output dim: 5, lower bound: -0.0006933, upper bound: 0.0006933

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.27 + 1.53 = 4.80 seconds
