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
0: (-0.0018075, -0.0010916, -0.0018075, -0.0010916, -0.0003892, 0.0003892)
1: (-0.0088975, -0.0070807, -0.0088975, -0.0070807, -0.0009876, 0.0009876)
2: (0.0295100, 0.0306372, 0.0295100, 0.0306372, -0.0006127, 0.0006127)
3: (0.0023552, 0.0044599, 0.0023552, 0.0044599, -0.0011441, 0.0011441)
4: (-0.0079432, -0.0060952, -0.0079432, -0.0060952, -0.0010046, 0.0010046)
5: (0.0107295, 0.0114295, 0.0107295, 0.0114295, -0.0003805, 0.0003805)
6: (0.0033772, 0.0060483, 0.0033772, 0.0060483, -0.0014520, 0.0014520)
7: (0.9804224, 0.9822915, 0.9804224, 0.9822915, -0.0010160, 0.0010160)
8: (-0.0075545, -0.0055505, -0.0075545, -0.0055505, -0.0010893, 0.0010893)
9: (-0.0013332, -0.0000095, -0.0013332, -0.0000095, -0.0007196, 0.0007196)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 1.64 = 3.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0006282, upper bound: 0.0006282

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0006251, upper bound: 0.0006251
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0006251, upper bound: 0.0006251
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 7, lower bound: -0.0006251, upper bound: 0.0006251
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 7, lower bound: -0.0006251, upper bound: 0.0006251

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0018075, -0.0010916, -0.0018075, -0.0010916, -0.0003892, 0.0003892
1: -0.0088975, -0.0070807, -0.0088975, -0.0070807, -0.0009878, 0.0009877
2: 0.0295100, 0.0306372, 0.0295100, 0.0306372, -0.0006128, 0.0006128
3: 0.0023552, 0.0044599, 0.0023552, 0.0044599, -0.0011442, 0.0011443
4: -0.0079432, -0.0060952, -0.0079432, -0.0060952, -0.0010047, 0.0010046
5: 0.0107295, 0.0114295, 0.0107295, 0.0114295, -0.0003806, 0.0003805
6: 0.0033772, 0.0060483, 0.0033772, 0.0060483, -0.0014521, 0.0014522
7: 0.9804224, 0.9822915, 0.9804224, 0.9822915, -0.0010161, 0.0010162
8: -0.0075545, -0.0055505, -0.0075545, -0.0055505, -0.0010894, 0.0010895
9: -0.0013332, -0.0000095, -0.0013332, -0.0000095, -0.0007197, 0.0007196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006144, upper bound: 0.0006048
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006046, upper bound: 0.0006144
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0018075, -0.0010916, -0.0018075, -0.0010916, -0.0003892, 0.0003892
1: -0.0088975, -0.0070807, -0.0088975, -0.0070807, -0.0009877, 0.0009878
2: 0.0295100, 0.0306372, 0.0295100, 0.0306372, -0.0006128, 0.0006128
3: 0.0023552, 0.0044599, 0.0023552, 0.0044599, -0.0011443, 0.0011442
4: -0.0079432, -0.0060952, -0.0079432, -0.0060952, -0.0010046, 0.0010047
5: 0.0107295, 0.0114295, 0.0107295, 0.0114295, -0.0003805, 0.0003806
6: 0.0033772, 0.0060483, 0.0033772, 0.0060483, -0.0014522, 0.0014521
7: 0.9804224, 0.9822915, 0.9804224, 0.9822915, -0.0010162, 0.0010161
8: -0.0075545, -0.0055505, -0.0075545, -0.0055505, -0.0010895, 0.0010894
9: -0.0013332, -0.0000095, -0.0013332, -0.0000095, -0.0007196, 0.0007197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005359, upper bound: 0.0005358
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0005358, upper bound: 0.0005359
time: 0.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.68
Output dim: 7, lower bound: -0.0006144, upper bound: 0.0006048
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.68
Output dim: 7, lower bound: -0.0006046, upper bound: 0.0006144
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.68
Output dim: 7, lower bound: -0.0005359, upper bound: 0.0005358
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.68
Output dim: 7, lower bound: -0.0005358, upper bound: 0.0005359

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.04 + 6.92 = 9.96 seconds
