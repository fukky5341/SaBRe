## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00073336


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0068329, 0.0083400, 0.0068329, 0.0083400, -0.0007056, 0.0007056)
1: (0.0023095, 0.0025272, 0.0023095, 0.0025272, -0.0001019, 0.0001019)
2: (0.0097489, 0.0105821, 0.0097489, 0.0105821, -0.0003901, 0.0003901)
3: (-0.0045977, -0.0037359, -0.0045977, -0.0037359, -0.0004034, 0.0004034)
4: (0.0000074, 0.0009403, 0.0000074, 0.0009403, -0.0004367, 0.0004367)
5: (0.0032235, 0.0041064, 0.0032235, 0.0041064, -0.0004133, 0.0004133)
6: (-0.0095104, -0.0060075, -0.0095104, -0.0060075, -0.0016399, 0.0016399)
7: (0.0056250, 0.0103956, 0.0056250, 0.0103956, -0.0022334, 0.0022334)
8: (0.9931763, 0.9965367, 0.9931763, 0.9965367, -0.0015732, 0.0015732)
9: (-0.0127436, -0.0096931, -0.0127436, -0.0096931, -0.0014281, 0.0014281)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.78 + 1.37 = 3.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0011933, upper bound: 0.0011933

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010503, upper bound: 0.0011395
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011395, upper bound: 0.0010503
time: 0.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 8, lower bound: -0.0010503, upper bound: 0.0011395
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 8, lower bound: -0.0011395, upper bound: 0.0010503

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068329, 0.0083400, 0.0068329, 0.0083400, -0.0006669, 0.0006961
1: 0.0023095, 0.0025272, 0.0023095, 0.0025272, -0.0000963, 0.0001006
2: 0.0097489, 0.0105821, 0.0097489, 0.0105821, -0.0003848, 0.0003687
3: -0.0045977, -0.0037359, -0.0045977, -0.0037359, -0.0003980, 0.0003813
4: 0.0000074, 0.0009403, 0.0000074, 0.0009403, -0.0004128, 0.0004309
5: 0.0032235, 0.0041064, 0.0032235, 0.0041064, -0.0004078, 0.0003907
6: -0.0095104, -0.0060075, -0.0095104, -0.0060075, -0.0016179, 0.0015501
7: 0.0056250, 0.0103956, 0.0056250, 0.0103956, -0.0021111, 0.0022034
8: 0.9931763, 0.9965367, 0.9931763, 0.9965367, -0.0014871, 0.0015521
9: -0.0127436, -0.0096931, -0.0127436, -0.0096931, -0.0014089, 0.0013499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008669, upper bound: 0.0008691
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008669, upper bound: 0.0008691
time: 0.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068329, 0.0083400, 0.0068329, 0.0083400, -0.0007056, 0.0006669
1: 0.0023095, 0.0025272, 0.0023095, 0.0025272, -0.0001019, 0.0000963
2: 0.0097489, 0.0105821, 0.0097489, 0.0105821, -0.0003687, 0.0003901
3: -0.0045977, -0.0037359, -0.0045977, -0.0037359, -0.0003813, 0.0004034
4: 0.0000074, 0.0009403, 0.0000074, 0.0009403, -0.0004367, 0.0004128
5: 0.0032235, 0.0041064, 0.0032235, 0.0041064, -0.0003907, 0.0004133
6: -0.0095104, -0.0060075, -0.0095104, -0.0060075, -0.0015501, 0.0016399
7: 0.0056250, 0.0103956, 0.0056250, 0.0103956, -0.0022334, 0.0021111
8: 0.9931763, 0.9965367, 0.9931763, 0.9965367, -0.0015732, 0.0014871
9: -0.0127436, -0.0096931, -0.0127436, -0.0096931, -0.0013499, 0.0014281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008691, upper bound: 0.0008669
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008691, upper bound: 0.0008669
time: 0.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 8, lower bound: -0.0008669, upper bound: 0.0008691
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 8, lower bound: -0.0008669, upper bound: 0.0008691
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 8, lower bound: -0.0008691, upper bound: 0.0008669
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 8, lower bound: -0.0008691, upper bound: 0.0008669

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068329, 0.0083400, 0.0068329, 0.0083400, -0.0006308, 0.0006594
1: 0.0023095, 0.0025272, 0.0023095, 0.0025272, -0.0000911, 0.0000953
2: 0.0097489, 0.0105821, 0.0097489, 0.0105821, -0.0003646, 0.0003488
3: -0.0045977, -0.0037359, -0.0045977, -0.0037359, -0.0003771, 0.0003607
4: 0.0000074, 0.0009403, 0.0000074, 0.0009403, -0.0003905, 0.0004082
5: 0.0032235, 0.0041064, 0.0032235, 0.0041064, -0.0003863, 0.0003695
6: -0.0095104, -0.0060075, -0.0095104, -0.0060075, -0.0015327, 0.0014662
7: 0.0056250, 0.0103956, 0.0056250, 0.0103956, -0.0019969, 0.0020874
8: 0.9931763, 0.9965367, 0.9931763, 0.9965367, -0.0014066, 0.0014704
9: -0.0127436, -0.0096931, -0.0127436, -0.0096931, -0.0013347, 0.0012769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068329, 0.0083400, 0.0068329, 0.0083400, -0.0006303, 0.0006961
1: 0.0023095, 0.0025272, 0.0023095, 0.0025272, -0.0000911, 0.0001006
2: 0.0097489, 0.0105821, 0.0097489, 0.0105821, -0.0003848, 0.0003484
3: -0.0045977, -0.0037359, -0.0045977, -0.0037359, -0.0003980, 0.0003604
4: 0.0000074, 0.0009403, 0.0000074, 0.0009403, -0.0003901, 0.0004309
5: 0.0032235, 0.0041064, 0.0032235, 0.0041064, -0.0004078, 0.0003692
6: -0.0095104, -0.0060075, -0.0095104, -0.0060075, -0.0016179, 0.0014649
7: 0.0056250, 0.0103956, 0.0056250, 0.0103956, -0.0019950, 0.0022034
8: 0.9931763, 0.9965367, 0.9931763, 0.9965367, -0.0014053, 0.0015521
9: -0.0127436, -0.0096931, -0.0127436, -0.0096931, -0.0014089, 0.0012757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068329, 0.0083400, 0.0068329, 0.0083400, -0.0006695, 0.0006303
1: 0.0023095, 0.0025272, 0.0023095, 0.0025272, -0.0000967, 0.0000911
2: 0.0097489, 0.0105821, 0.0097489, 0.0105821, -0.0003484, 0.0003701
3: -0.0045977, -0.0037359, -0.0045977, -0.0037359, -0.0003604, 0.0003828
4: 0.0000074, 0.0009403, 0.0000074, 0.0009403, -0.0004144, 0.0003901
5: 0.0032235, 0.0041064, 0.0032235, 0.0041064, -0.0003692, 0.0003922
6: -0.0095104, -0.0060075, -0.0095104, -0.0060075, -0.0014649, 0.0015561
7: 0.0056250, 0.0103956, 0.0056250, 0.0103956, -0.0021192, 0.0019950
8: 0.9931763, 0.9965367, 0.9931763, 0.9965367, -0.0014928, 0.0014053
9: -0.0127436, -0.0096931, -0.0127436, -0.0096931, -0.0012757, 0.0013551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068329, 0.0083400, 0.0068329, 0.0083400, -0.0006689, 0.0006669
1: 0.0023095, 0.0025272, 0.0023095, 0.0025272, -0.0000966, 0.0000963
2: 0.0097489, 0.0105821, 0.0097489, 0.0105821, -0.0003687, 0.0003698
3: -0.0045977, -0.0037359, -0.0045977, -0.0037359, -0.0003813, 0.0003825
4: 0.0000074, 0.0009403, 0.0000074, 0.0009403, -0.0004141, 0.0004128
5: 0.0032235, 0.0041064, 0.0032235, 0.0041064, -0.0003907, 0.0003918
6: -0.0095104, -0.0060075, -0.0095104, -0.0060075, -0.0015501, 0.0015547
7: 0.0056250, 0.0103956, 0.0056250, 0.0103956, -0.0021174, 0.0021111
8: 0.9931763, 0.9965367, 0.9931763, 0.9965367, -0.0014915, 0.0014871
9: -0.0127436, -0.0096931, -0.0127436, -0.0096931, -0.0013499, 0.0013539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
time: 0.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007015, upper bound: 0.0007015

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.15 + 17.62 = 20.77 seconds
