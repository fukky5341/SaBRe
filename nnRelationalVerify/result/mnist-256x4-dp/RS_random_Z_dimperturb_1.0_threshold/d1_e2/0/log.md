## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0029930225


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872)
1: (0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107)
2: (-0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0057238, 0.0057238)
3: (0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783)
4: (0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503)
5: (0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701)
6: (-0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600)
7: (-0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067)
8: (-0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109517, 0.0109517)
9: (-0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.11 + 3.02 = 4.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0032357, upper bound: 0.0032357

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027573, upper bound: 0.0027573
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027573, upper bound: 0.0027573
time: 1.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.60 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 4.60
Output dim: 1, lower bound: -0.0027573, upper bound: 0.0027573
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 4.60
Output dim: 1, lower bound: -0.0027573, upper bound: 0.0027573

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 4.13 + 4.60 = 8.73 seconds
