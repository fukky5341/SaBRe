## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00035784


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043172, -0.0042600, -0.0043172, -0.0042600, -0.0000280, 0.0000280)
1: (0.0033355, 0.0036525, 0.0033355, 0.0036525, -0.0001550, 0.0001550)
2: (0.0068062, 0.0075143, 0.0068062, 0.0075143, -0.0003464, 0.0003464)
3: (0.0041678, 0.0044662, 0.0041678, 0.0044662, -0.0001460, 0.0001460)
4: (1.0129198, 1.0140774, 1.0129198, 1.0140774, -0.0005663, 0.0005663)
5: (0.0047548, 0.0049800, 0.0047548, 0.0049800, -0.0001102, 0.0001102)
6: (-0.0122237, -0.0119306, -0.0122237, -0.0119306, -0.0001434, 0.0001434)
7: (-0.0103626, -0.0103252, -0.0103626, -0.0103252, -0.0000183, 0.0000183)
8: (-0.0026110, -0.0024085, -0.0026110, -0.0024085, -0.0000991, 0.0000991)
9: (-0.0061135, -0.0050997, -0.0061135, -0.0050997, -0.0004959, 0.0004959)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 1.21 = 2.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0003588, upper bound: 0.0003589

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 131
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 85

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0003526, upper bound: 0.0003282
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0003282, upper bound: 0.0003526
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.80 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.80
Output dim: 4, lower bound: -0.0003526, upper bound: 0.0003282
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.80
Output dim: 4, lower bound: -0.0003282, upper bound: 0.0003526

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.53 + 0.80 = 3.33 seconds
