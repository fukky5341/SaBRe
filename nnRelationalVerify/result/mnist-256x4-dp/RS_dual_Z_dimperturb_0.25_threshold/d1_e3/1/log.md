## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00018112


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040997, -0.0040850, -0.0040997, -0.0040850, -0.0000068, 0.0000068)
1: (-0.0061905, -0.0056381, -0.0061905, -0.0056381, -0.0002542, 0.0002542)
2: (0.9690347, 0.9696975, 0.9690347, 0.9696975, -0.0003050, 0.0003050)
3: (0.0179106, 0.0228000, 0.0179106, 0.0228000, -0.0022499, 0.0022499)
4: (-0.0024271, -0.0020552, -0.0024271, -0.0020552, -0.0001711, 0.0001711)
5: (0.0148173, 0.0151932, 0.0148173, 0.0151932, -0.0001729, 0.0001729)
6: (0.0045168, 0.0046996, 0.0045168, 0.0046996, -0.0000841, 0.0000841)
7: (-0.0136871, -0.0124199, -0.0136871, -0.0124199, -0.0005831, 0.0005831)
8: (0.0058705, 0.0068758, 0.0058705, 0.0068758, -0.0004626, 0.0004626)
9: (0.0082832, 0.0100913, 0.0082832, 0.0100913, -0.0008320, 0.0008320)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 1.35 = 2.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0002259, upper bound: 0.0002259

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002147, upper bound: 0.0002176
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002176, upper bound: 0.0002147
time: 0.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 2, lower bound: -0.0002147, upper bound: 0.0002176
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 2, lower bound: -0.0002176, upper bound: 0.0002147

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040997, -0.0040850, -0.0040997, -0.0040850, -0.0000063, 0.0000064
1: -0.0061905, -0.0056381, -0.0061905, -0.0056381, -0.0002376, 0.0002387
2: 0.9690347, 0.9696975, 0.9690347, 0.9696975, -0.0002851, 0.0002864
3: 0.0179106, 0.0228000, 0.0179106, 0.0228000, -0.0021027, 0.0021126
4: -0.0024271, -0.0020552, -0.0024271, -0.0020552, -0.0001607, 0.0001599
5: 0.0148173, 0.0151932, 0.0148173, 0.0151932, -0.0001624, 0.0001616
6: 0.0045168, 0.0046996, 0.0045168, 0.0046996, -0.0000786, 0.0000790
7: -0.0136871, -0.0124199, -0.0136871, -0.0124199, -0.0005475, 0.0005449
8: 0.0058705, 0.0068758, 0.0058705, 0.0068758, -0.0004344, 0.0004323
9: 0.0082832, 0.0100913, 0.0082832, 0.0100913, -0.0007812, 0.0007776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002045, upper bound: 0.0002113
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002072, upper bound: 0.0002053
time: 0.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040997, -0.0040850, -0.0040997, -0.0040850, -0.0000064, 0.0000063
1: -0.0061905, -0.0056381, -0.0061905, -0.0056381, -0.0002387, 0.0002376
2: 0.9690347, 0.9696975, 0.9690347, 0.9696975, -0.0002864, 0.0002851
3: 0.0179106, 0.0228000, 0.0179106, 0.0228000, -0.0021126, 0.0021027
4: -0.0024271, -0.0020552, -0.0024271, -0.0020552, -0.0001599, 0.0001607
5: 0.0148173, 0.0151932, 0.0148173, 0.0151932, -0.0001616, 0.0001624
6: 0.0045168, 0.0046996, 0.0045168, 0.0046996, -0.0000790, 0.0000786
7: -0.0136871, -0.0124199, -0.0136871, -0.0124199, -0.0005449, 0.0005475
8: 0.0058705, 0.0068758, 0.0058705, 0.0068758, -0.0004323, 0.0004344
9: 0.0082832, 0.0100913, 0.0082832, 0.0100913, -0.0007776, 0.0007812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002054, upper bound: 0.0002072
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002113, upper bound: 0.0002045
time: 0.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 2, lower bound: -0.0002045, upper bound: 0.0002113
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 2, lower bound: -0.0002072, upper bound: 0.0002053
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 2, lower bound: -0.0002054, upper bound: 0.0002072
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 2, lower bound: -0.0002113, upper bound: 0.0002045

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040997, -0.0040850, -0.0040997, -0.0040850, -0.0000055, 0.0000057
1: -0.0061905, -0.0056381, -0.0061905, -0.0056381, -0.0002068, 0.0002118
2: 0.9690347, 0.9696975, 0.9690347, 0.9696975, -0.0002482, 0.0002542
3: 0.0179106, 0.0228000, 0.0179106, 0.0228000, -0.0018305, 0.0018749
4: -0.0024271, -0.0020552, -0.0024271, -0.0020552, -0.0001426, 0.0001392
5: 0.0148173, 0.0151932, 0.0148173, 0.0151932, -0.0001441, 0.0001407
6: 0.0045168, 0.0046996, 0.0045168, 0.0046996, -0.0000684, 0.0000701
7: -0.0136871, -0.0124199, -0.0136871, -0.0124199, -0.0004859, 0.0004744
8: 0.0058705, 0.0068758, 0.0058705, 0.0068758, -0.0003855, 0.0003764
9: 0.0082832, 0.0100913, 0.0082832, 0.0100913, -0.0006933, 0.0006769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001608, upper bound: 0.0001665
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001608, upper bound: 0.0001665
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040997, -0.0040850, -0.0040997, -0.0040850, -0.0000056, 0.0000056
1: -0.0061905, -0.0056381, -0.0061905, -0.0056381, -0.0002103, 0.0002079
2: 0.9690347, 0.9696975, 0.9690347, 0.9696975, -0.0002524, 0.0002495
3: 0.0179106, 0.0228000, 0.0179106, 0.0228000, -0.0018616, 0.0018404
4: -0.0024271, -0.0020552, -0.0024271, -0.0020552, -0.0001400, 0.0001416
5: 0.0148173, 0.0151932, 0.0148173, 0.0151932, -0.0001415, 0.0001431
6: 0.0045168, 0.0046996, 0.0045168, 0.0046996, -0.0000696, 0.0000688
7: -0.0136871, -0.0124199, -0.0136871, -0.0124199, -0.0004770, 0.0004825
8: 0.0058705, 0.0068758, 0.0058705, 0.0068758, -0.0003784, 0.0003828
9: 0.0082832, 0.0100913, 0.0082832, 0.0100913, -0.0006806, 0.0006884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001660, upper bound: 0.0001614
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001660, upper bound: 0.0001614
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040997, -0.0040850, -0.0040997, -0.0040850, -0.0000056, 0.0000056
1: -0.0061905, -0.0056381, -0.0061905, -0.0056381, -0.0002079, 0.0002103
2: 0.9690347, 0.9696975, 0.9690347, 0.9696975, -0.0002495, 0.0002524
3: 0.0179106, 0.0228000, 0.0179106, 0.0228000, -0.0018404, 0.0018616
4: -0.0024271, -0.0020552, -0.0024271, -0.0020552, -0.0001416, 0.0001400
5: 0.0148173, 0.0151932, 0.0148173, 0.0151932, -0.0001431, 0.0001415
6: 0.0045168, 0.0046996, 0.0045168, 0.0046996, -0.0000688, 0.0000696
7: -0.0136871, -0.0124199, -0.0136871, -0.0124199, -0.0004825, 0.0004770
8: 0.0058705, 0.0068758, 0.0058705, 0.0068758, -0.0003828, 0.0003784
9: 0.0082832, 0.0100913, 0.0082832, 0.0100913, -0.0006884, 0.0006806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001614, upper bound: 0.0001660
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001614, upper bound: 0.0001660
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040997, -0.0040850, -0.0040997, -0.0040850, -0.0000057, 0.0000055
1: -0.0061905, -0.0056381, -0.0061905, -0.0056381, -0.0002118, 0.0002068
2: 0.9690347, 0.9696975, 0.9690347, 0.9696975, -0.0002542, 0.0002482
3: 0.0179106, 0.0228000, 0.0179106, 0.0228000, -0.0018749, 0.0018305
4: -0.0024271, -0.0020552, -0.0024271, -0.0020552, -0.0001392, 0.0001426
5: 0.0148173, 0.0151932, 0.0148173, 0.0151932, -0.0001407, 0.0001441
6: 0.0045168, 0.0046996, 0.0045168, 0.0046996, -0.0000701, 0.0000684
7: -0.0136871, -0.0124199, -0.0136871, -0.0124199, -0.0004744, 0.0004859
8: 0.0058705, 0.0068758, 0.0058705, 0.0068758, -0.0003764, 0.0003855
9: 0.0082832, 0.0100913, 0.0082832, 0.0100913, -0.0006769, 0.0006933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001665, upper bound: 0.0001608
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001665, upper bound: 0.0001608
time: 0.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 2, lower bound: -0.0001608, upper bound: 0.0001665
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 2, lower bound: -0.0001608, upper bound: 0.0001665
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 2, lower bound: -0.0001660, upper bound: 0.0001614
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 2, lower bound: -0.0001660, upper bound: 0.0001614
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 2, lower bound: -0.0001614, upper bound: 0.0001660
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 2, lower bound: -0.0001614, upper bound: 0.0001660
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 2, lower bound: -0.0001665, upper bound: 0.0001608
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 2, lower bound: -0.0001665, upper bound: 0.0001608

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.64 + 15.32 = 17.95 seconds
