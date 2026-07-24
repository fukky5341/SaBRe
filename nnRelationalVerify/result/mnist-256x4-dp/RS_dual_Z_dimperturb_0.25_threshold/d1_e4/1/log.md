## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00048692


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0033130, 0.0042872, 0.0033130, 0.0042872, -0.0005246, 0.0005246)
1: (0.0018009, 0.0019417, 0.0018009, 0.0019417, -0.0000758, 0.0000758)
2: (0.0119896, 0.0125282, 0.0119896, 0.0125282, -0.0002901, 0.0002901)
3: (-0.0022803, -0.0017232, -0.0022803, -0.0017232, -0.0003000, 0.0003000)
4: (-0.0021714, -0.0015684, -0.0021714, -0.0015684, -0.0003248, 0.0003248)
5: (0.0055976, 0.0061683, 0.0055976, 0.0061683, -0.0003073, 0.0003073)
6: (-0.0000907, 0.0021736, -0.0000907, 0.0021736, -0.0012194, 0.0012194)
7: (-0.0055169, -0.0024332, -0.0055169, -0.0024332, -0.0016607, 0.0016607)
8: (0.9853277, 0.9874998, 0.9853277, 0.9874998, -0.0011698, 0.0011698)
9: (-0.0045405, -0.0025687, -0.0045405, -0.0025687, -0.0010619, 0.0010619)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 1.40 = 2.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0006922, upper bound: 0.0006922

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006295, upper bound: 0.0006327
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006327, upper bound: 0.0006295
time: 0.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 8, lower bound: -0.0006295, upper bound: 0.0006327
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 8, lower bound: -0.0006327, upper bound: 0.0006295

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0033130, 0.0042872, 0.0033130, 0.0042872, -0.0004641, 0.0004577
1: 0.0018009, 0.0019417, 0.0018009, 0.0019417, -0.0000671, 0.0000661
2: 0.0119896, 0.0125282, 0.0119896, 0.0125282, -0.0002530, 0.0002566
3: -0.0022803, -0.0017232, -0.0022803, -0.0017232, -0.0002617, 0.0002654
4: -0.0021714, -0.0015684, -0.0021714, -0.0015684, -0.0002873, 0.0002833
5: 0.0055976, 0.0061683, 0.0055976, 0.0061683, -0.0002681, 0.0002719
6: -0.0000907, 0.0021736, -0.0000907, 0.0021736, -0.0010637, 0.0010788
7: -0.0055169, -0.0024332, -0.0055169, -0.0024332, -0.0014692, 0.0014487
8: 0.9853277, 0.9874998, 0.9853277, 0.9874998, -0.0010350, 0.0010205
9: -0.0045405, -0.0025687, -0.0045405, -0.0025687, -0.0009263, 0.0009395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0003780, upper bound: 0.0003780
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0003780, upper bound: 0.0003780
time: 0.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0033130, 0.0042872, 0.0033130, 0.0042872, -0.0004577, 0.0004641
1: 0.0018009, 0.0019417, 0.0018009, 0.0019417, -0.0000661, 0.0000671
2: 0.0119896, 0.0125282, 0.0119896, 0.0125282, -0.0002566, 0.0002530
3: -0.0022803, -0.0017232, -0.0022803, -0.0017232, -0.0002654, 0.0002617
4: -0.0021714, -0.0015684, -0.0021714, -0.0015684, -0.0002833, 0.0002873
5: 0.0055976, 0.0061683, 0.0055976, 0.0061683, -0.0002719, 0.0002681
6: -0.0000907, 0.0021736, -0.0000907, 0.0021736, -0.0010788, 0.0010637
7: -0.0055169, -0.0024332, -0.0055169, -0.0024332, -0.0014487, 0.0014692
8: 0.9853277, 0.9874998, 0.9853277, 0.9874998, -0.0010205, 0.0010350
9: -0.0045405, -0.0025687, -0.0045405, -0.0025687, -0.0009395, 0.0009263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0003780, upper bound: 0.0003780
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0003780, upper bound: 0.0003780
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.45
Output dim: 8, lower bound: -0.0003780, upper bound: 0.0003780
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.45
Output dim: 8, lower bound: -0.0003780, upper bound: 0.0003780
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.45
Output dim: 8, lower bound: -0.0003780, upper bound: 0.0003780
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.45
Output dim: 8, lower bound: -0.0003780, upper bound: 0.0003780

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.87 + 6.13 = 9.00 seconds
