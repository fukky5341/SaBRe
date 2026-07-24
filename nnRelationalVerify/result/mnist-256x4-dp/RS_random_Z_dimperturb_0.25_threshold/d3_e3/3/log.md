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
Threshold: 0.000229875


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0130413, -0.0113299, -0.0130413, -0.0113299, -0.0010918, 0.0010918)
1: (-0.0066155, -0.0061330, -0.0066155, -0.0061330, -0.0003078, 0.0003078)
2: (-0.0102507, -0.0066906, -0.0102507, -0.0066906, -0.0022712, 0.0022712)
3: (0.0002708, 0.0007419, 0.0002708, 0.0007419, -0.0003006, 0.0003006)
4: (0.0110920, 0.0137526, 0.0110920, 0.0137526, -0.0016974, 0.0016974)
5: (0.9985879, 0.9993271, 0.9985879, 0.9993271, -0.0004716, 0.0004716)
6: (0.0066019, 0.0072729, 0.0066019, 0.0072729, -0.0004281, 0.0004281)
7: (0.0012557, 0.0037596, 0.0012557, 0.0037596, -0.0015974, 0.0015974)
8: (-0.0121190, -0.0101702, -0.0121190, -0.0101702, -0.0012433, 0.0012433)
9: (-0.0031323, -0.0029642, -0.0031323, -0.0029642, -0.0001073, 0.0001073)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.87 + 1.43 = 3.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0002900, upper bound: 0.0002900

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002842, upper bound: 0.0002839
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002837, upper bound: 0.0002842
time: 0.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 5, lower bound: -0.0002842, upper bound: 0.0002839
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 5, lower bound: -0.0002837, upper bound: 0.0002842

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0130413, -0.0113299, -0.0130413, -0.0113299, -0.0010597, 0.0010594
1: -0.0066155, -0.0061330, -0.0066155, -0.0061330, -0.0002988, 0.0002987
2: -0.0102507, -0.0066906, -0.0102507, -0.0066906, -0.0022045, 0.0022038
3: 0.0002708, 0.0007419, 0.0002708, 0.0007419, -0.0002917, 0.0002916
4: 0.0110920, 0.0137526, 0.0110920, 0.0137526, -0.0016470, 0.0016475
5: 0.9985879, 0.9993271, 0.9985879, 0.9993271, -0.0004576, 0.0004577
6: 0.0066019, 0.0072729, 0.0066019, 0.0072729, -0.0004153, 0.0004155
7: 0.0012557, 0.0037596, 0.0012557, 0.0037596, -0.0015500, 0.0015505
8: -0.0121190, -0.0101702, -0.0121190, -0.0101702, -0.0012067, 0.0012064
9: -0.0031323, -0.0029642, -0.0031323, -0.0029642, -0.0001041, 0.0001041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001666, upper bound: 0.0001655
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001666, upper bound: 0.0001655
time: 0.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0130413, -0.0113299, -0.0130413, -0.0113299, -0.0010594, 0.0010597
1: -0.0066155, -0.0061330, -0.0066155, -0.0061330, -0.0002987, 0.0002988
2: -0.0102507, -0.0066906, -0.0102507, -0.0066906, -0.0022038, 0.0022045
3: 0.0002708, 0.0007419, 0.0002708, 0.0007419, -0.0002916, 0.0002917
4: 0.0110920, 0.0137526, 0.0110920, 0.0137526, -0.0016475, 0.0016470
5: 0.9985879, 0.9993271, 0.9985879, 0.9993271, -0.0004577, 0.0004576
6: 0.0066019, 0.0072729, 0.0066019, 0.0072729, -0.0004155, 0.0004153
7: 0.0012557, 0.0037596, 0.0012557, 0.0037596, -0.0015505, 0.0015500
8: -0.0121190, -0.0101702, -0.0121190, -0.0101702, -0.0012064, 0.0012067
9: -0.0031323, -0.0029642, -0.0031323, -0.0029642, -0.0001041, 0.0001041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001655, upper bound: 0.0001666
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001655, upper bound: 0.0001666
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.84
Output dim: 5, lower bound: -0.0001666, upper bound: 0.0001655
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.84
Output dim: 5, lower bound: -0.0001666, upper bound: 0.0001655
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.84
Output dim: 5, lower bound: -0.0001655, upper bound: 0.0001666
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.84
Output dim: 5, lower bound: -0.0001655, upper bound: 0.0001666

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.31 + 6.80 = 10.10 seconds
