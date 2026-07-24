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
Threshold: 0.00079287


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0017615, -0.0000139, -0.0017615, -0.0000139, -0.0014531, 0.0014531)
1: (-0.0043571, -0.0037779, -0.0043571, -0.0037779, -0.0005134, 0.0005134)
2: (0.0123020, 0.0146562, 0.0123020, 0.0146562, -0.0019091, 0.0019091)
3: (1.0078764, 1.0093555, 1.0078764, 1.0093555, -0.0014791, 0.0014791)
4: (-0.0039963, -0.0036085, -0.0039963, -0.0036085, -0.0003066, 0.0003066)
5: (0.0025997, 0.0039471, 0.0025997, 0.0039471, -0.0011163, 0.0011163)
6: (-0.0024924, -0.0023550, -0.0024924, -0.0023550, -0.0001374, 0.0001374)
7: (-0.0129942, -0.0107451, -0.0129942, -0.0107451, -0.0022150, 0.0022150)
8: (-0.0107282, -0.0065105, -0.0107282, -0.0065105, -0.0032885, 0.0032885)
9: (-0.0010694, 0.0010292, -0.0010694, 0.0010292, -0.0016130, 0.0016130)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.74 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0008555, upper bound: 0.0008555

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007648, upper bound: 0.0007648
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007648, upper bound: 0.0007648
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.41 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.41
Output dim: 3, lower bound: -0.0007648, upper bound: 0.0007648
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.41
Output dim: 3, lower bound: -0.0007648, upper bound: 0.0007648

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.29 + 1.41 = 4.70 seconds
