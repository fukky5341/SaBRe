## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.7875192


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813)
1: (-0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893)
2: (-0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501)
3: (0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201)
4: (-0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563)
5: (-0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932)
6: (-0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614)
7: (-0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642)
8: (-0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244)
9: (-0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 2.77 = 4.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7378560, upper bound: 0.7378560
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7378560, upper bound: 0.7378560
time: 1.05 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.11 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 2.11
Output dim: 3, lower bound: -0.7378560, upper bound: 0.7378560
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 2.11
Output dim: 3, lower bound: -0.7378560, upper bound: 0.7378560

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 4.10 + 2.11 = 6.21 seconds
