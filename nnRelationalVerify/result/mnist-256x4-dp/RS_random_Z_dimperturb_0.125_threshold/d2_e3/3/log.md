## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.432e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0081533, -0.0076136, -0.0081533, -0.0076136, -0.0002582, 0.0002582)
1: (-0.0052374, -0.0050852, -0.0052374, -0.0050852, -0.0000728, 0.0000728)
2: (-0.0000828, 0.0010401, -0.0000828, 0.0010401, -0.0005371, 0.0005371)
3: (0.0016163, 0.0017649, 0.0016163, 0.0017649, -0.0000711, 0.0000711)
4: (0.0053146, 0.0061537, 0.0053146, 0.0061537, -0.0004014, 0.0004014)
5: (0.9969828, 0.9972159, 0.9969828, 0.9972159, -0.0001115, 0.0001115)
6: (0.0051449, 0.0053565, 0.0051449, 0.0053565, -0.0001012, 0.0001012)
7: (-0.0041815, -0.0033918, -0.0041815, -0.0033918, -0.0003778, 0.0003778)
8: (-0.0065530, -0.0059384, -0.0065530, -0.0059384, -0.0002940, 0.0002940)
9: (-0.0034974, -0.0034444, -0.0034974, -0.0034444, -0.0000254, 0.0000254)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 1.25 = 2.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0000753, upper bound: 0.0000757

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000542, upper bound: 0.0000542
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000542, upper bound: 0.0000542
time: 0.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.89 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.89
Output dim: 5, lower bound: -0.0000542, upper bound: 0.0000542
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.89
Output dim: 5, lower bound: -0.0000542, upper bound: 0.0000542

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.73 + 0.89 = 3.62 seconds
