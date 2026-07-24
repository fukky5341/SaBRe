## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00399952


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0021738, 0.0021738)
1: (-0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0015875, 0.0015875)
2: (0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006964, 0.0006964)
3: (0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0010446, 0.0010446)
4: (-0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0013326, 0.0013326)
5: (-0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009674, 0.0009674)
6: (-0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0023244, 0.0023244)
7: (-0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0077156, 0.0077156)
8: (0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0073751, 0.0073751)
9: (0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0051094, 0.0051094)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.63 = 3.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0050089, upper bound: 0.0050089

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048747, upper bound: 0.0048857
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048857, upper bound: 0.0048747
time: 0.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 8, lower bound: -0.0048747, upper bound: 0.0048857
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 8, lower bound: -0.0048857, upper bound: 0.0048747

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0021517, 0.0021487
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0015826, 0.0015813
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006869, 0.0006877
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009788, 0.0009798
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0012632, 0.0012601
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009656, 0.0009660
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0021490, 0.0021461
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0073302, 0.0073121
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0070833, 0.0070758
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0048550, 0.0048664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047164, upper bound: 0.0048781
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048670, upper bound: 0.0047171
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0021487, 0.0021517
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0015813, 0.0015826
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006877, 0.0006869
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009798, 0.0009788
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0012601, 0.0012632
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009660, 0.0009656
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0021461, 0.0021490
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0073121, 0.0073302
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0070758, 0.0070833
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0048664, 0.0048550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047171, upper bound: 0.0048670
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048781, upper bound: 0.0047164
time: 0.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 8, lower bound: -0.0047164, upper bound: 0.0048781
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 8, lower bound: -0.0048670, upper bound: 0.0047171
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 8, lower bound: -0.0047171, upper bound: 0.0048670
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 8, lower bound: -0.0048781, upper bound: 0.0047164

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0020514, 0.0020411
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0015266, 0.0015257
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006495, 0.0006534
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009637, 0.0009614
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011145, 0.0011290
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009377, 0.0009373
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019770, 0.0019485
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064749, 0.0065588
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0063320, 0.0064138
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0043623, 0.0043065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046762, upper bound: 0.0048403
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046730, upper bound: 0.0048405
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0020442, 0.0020494
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0015270, 0.0015256
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006528, 0.0006503
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009604, 0.0009641
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011283, 0.0011114
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009368, 0.0009382
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019514, 0.0019693
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0065534, 0.0064568
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0064054, 0.0063245
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0042951, 0.0043573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048291, upper bound: 0.0046729
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048290, upper bound: 0.0046762
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0020494, 0.0020442
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0015256, 0.0015270
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006503, 0.0006528
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009641, 0.0009604
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011114, 0.0011283
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009382, 0.0009368
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019693, 0.0019514
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064568, 0.0065534
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0063245, 0.0064054
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0043573, 0.0042951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046762, upper bound: 0.0048290
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046729, upper bound: 0.0048291
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0020411, 0.0020514
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0015257, 0.0015266
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006534, 0.0006495
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009614, 0.0009637
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011290, 0.0011145
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009373, 0.0009377
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019485, 0.0019770
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0065588, 0.0064749
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0064138, 0.0063320
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0043065, 0.0043623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048405, upper bound: 0.0046730
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048403, upper bound: 0.0046762
time: 0.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.60 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 8, lower bound: -0.0046762, upper bound: 0.0048403
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 8, lower bound: -0.0046730, upper bound: 0.0048405
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 8, lower bound: -0.0048291, upper bound: 0.0046729
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 8, lower bound: -0.0048290, upper bound: 0.0046762
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 8, lower bound: -0.0046762, upper bound: 0.0048290
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 8, lower bound: -0.0046729, upper bound: 0.0048291
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 8, lower bound: -0.0048405, upper bound: 0.0046730
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 8, lower bound: -0.0048403, upper bound: 0.0046762

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019854, 0.0020013
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014734, 0.0014928
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006369, 0.0006329
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009413, 0.0009289
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011064, 0.0011180
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009169, 0.0009037
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019431, 0.0019127
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064236, 0.0064843
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0062498, 0.0062927
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0043056, 0.0042682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041507, upper bound: 0.0042982
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041460, upper bound: 0.0043027
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0020096, 0.0019751
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014938, 0.0014725
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006290, 0.0006405
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009311, 0.0009376
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011035, 0.0011192
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009041, 0.0009166
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019412, 0.0019138
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064004, 0.0064970
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0062109, 0.0063240
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0043185, 0.0042498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044908, upper bound: 0.0046908
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045098, upper bound: 0.0046536
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019781, 0.0020070
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014738, 0.0014923
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006395, 0.0006298
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009370, 0.0009316
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011186, 0.0011004
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009160, 0.0009046
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019171, 0.0019335
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064922, 0.0063823
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0063190, 0.0062034
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0042384, 0.0043138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046434, upper bound: 0.0045098
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046771, upper bound: 0.0044908
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0020040, 0.0019833
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014941, 0.0014724
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006323, 0.0006377
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009279, 0.0009411
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011173, 0.0011031
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009033, 0.0009172
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019157, 0.0019360
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064788, 0.0064048
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0062843, 0.0062419
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0042564, 0.0043005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046434, upper bound: 0.0045145
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046769, upper bound: 0.0044959
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019833, 0.0020040
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014724, 0.0014941
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006377, 0.0006323
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009411, 0.0009279
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011031, 0.0011173
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009172, 0.0009033
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019360, 0.0019157
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064048, 0.0064788
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0062419, 0.0062843
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0043005, 0.0042564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044959, upper bound: 0.0046769
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045145, upper bound: 0.0046434
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0020070, 0.0019781
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014923, 0.0014738
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006298, 0.0006395
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009316, 0.0009370
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011004, 0.0011186
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009046, 0.0009160
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019335, 0.0019171
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0063823, 0.0064922
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0062034, 0.0063190
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0043138, 0.0042384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044908, upper bound: 0.0046771
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045098, upper bound: 0.0046434
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019751, 0.0020096
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014725, 0.0014938
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006405, 0.0006290
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009376, 0.0009311
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011192, 0.0011035
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009166, 0.0009041
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019138, 0.0019412
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064970, 0.0064004
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0063240, 0.0062109
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0042498, 0.0043185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043027, upper bound: 0.0041458
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042982, upper bound: 0.0041504
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0020013, 0.0019854
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014928, 0.0014734
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006329, 0.0006369
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009289, 0.0009413
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011180, 0.0011064
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009037, 0.0009169
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019127, 0.0019431
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064843, 0.0064236
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0062927, 0.0062498
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0042682, 0.0043056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046536, upper bound: 0.0045143
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046902, upper bound: 0.0044954
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0041507, upper bound: 0.0042982
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0041460, upper bound: 0.0043027
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0044908, upper bound: 0.0046908
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0045098, upper bound: 0.0046536
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0046434, upper bound: 0.0045098
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0046771, upper bound: 0.0044908
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0046434, upper bound: 0.0045145
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0046769, upper bound: 0.0044959
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0044959, upper bound: 0.0046769
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0045145, upper bound: 0.0046434
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0044908, upper bound: 0.0046771
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0045098, upper bound: 0.0046434
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0043027, upper bound: 0.0041458
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0042982, upper bound: 0.0041504
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0046536, upper bound: 0.0045143
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0046902, upper bound: 0.0044954

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019651, 0.0019951
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014581, 0.0014850
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006357, 0.0006265
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009303, 0.0009231
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011081, 0.0011119
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009114, 0.0008943
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019323, 0.0019096
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064349, 0.0064460
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0062726, 0.0062447
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0042780, 0.0042777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019854, 0.0019810
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014734, 0.0014775
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006305, 0.0006329
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009355, 0.0009289
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011003, 0.0011180
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009074, 0.0009037
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019400, 0.0019127
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0063854, 0.0064843
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0062018, 0.0062927
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0043056, 0.0042406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019700, 0.0019345
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014703, 0.0014485
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006153, 0.0006272
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009248, 0.0009309
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010649, 0.0010854
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0008908, 0.0009031
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018931, 0.0018632
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0061781, 0.0063029
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0060075, 0.0061433
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041902, 0.0041033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019691, 0.0019375
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014699, 0.0014489
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006166, 0.0006268
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009244, 0.0009314
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010666, 0.0010806
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0008908, 0.0009032
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018906, 0.0018665
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0061890, 0.0062747
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0060192, 0.0061205
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041720, 0.0041103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019389, 0.0019665
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014503, 0.0014684
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006259, 0.0006168
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009308, 0.0009249
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010800, 0.0010653
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009027, 0.0008913
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018701, 0.0018829
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0062699, 0.0061822
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0061156, 0.0060199
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041062, 0.0041673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039739, upper bound: 0.0038666
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039739, upper bound: 0.0038666
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019376, 0.0019670
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014499, 0.0014686
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006261, 0.0006161
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009303, 0.0009250
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010842, 0.0010618
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009025, 0.0008913
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018665, 0.0018860
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0062947, 0.0061600
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0061318, 0.0060000
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0040919, 0.0041831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039741, upper bound: 0.0038665
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039741, upper bound: 0.0038665
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019642, 0.0019428
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014699, 0.0014484
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006187, 0.0006243
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009213, 0.0009344
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010787, 0.0010676
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0008899, 0.0009035
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018681, 0.0018854
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0062566, 0.0062011
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0060809, 0.0060544
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041229, 0.0041540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039739, upper bound: 0.0038666
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039739, upper bound: 0.0038666
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019634, 0.0019454
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014702, 0.0014501
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006195, 0.0006241
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009212, 0.0009348
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010828, 0.0010645
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0008905, 0.0009039
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018651, 0.0018900
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0062804, 0.0061825
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0061014, 0.0060385
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041099, 0.0041702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039741, upper bound: 0.0038665
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039741, upper bound: 0.0038665
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019454, 0.0019634
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014501, 0.0014702
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006241, 0.0006195
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009348, 0.0009212
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010645, 0.0010828
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009039, 0.0008905
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018900, 0.0018651
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0061825, 0.0062804
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0060385, 0.0061014
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041702, 0.0041099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038665, upper bound: 0.0039741
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038665, upper bound: 0.0039741
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019428, 0.0019642
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014484, 0.0014699
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006243, 0.0006187
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009344, 0.0009213
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010676, 0.0010787
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009035, 0.0008899
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018854, 0.0018681
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0062011, 0.0062566
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0060544, 0.0060809
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041540, 0.0041229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038666, upper bound: 0.0039739
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038666, upper bound: 0.0039739
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019670, 0.0019376
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014686, 0.0014499
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006161, 0.0006261
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009250, 0.0009303
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010618, 0.0010842
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0008913, 0.0009025
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018860, 0.0018665
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0061600, 0.0062947
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0060000, 0.0061318
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041831, 0.0040919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038665, upper bound: 0.0039741
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038665, upper bound: 0.0039741
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019665, 0.0019389
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014684, 0.0014503
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006168, 0.0006259
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009249, 0.0009308
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010653, 0.0010800
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0008913, 0.0009027
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018829, 0.0018701
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0061822, 0.0062699
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0060199, 0.0061156
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041673, 0.0041062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038666, upper bound: 0.0039739
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038666, upper bound: 0.0039739
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019548, 0.0020053
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014572, 0.0014874
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006394, 0.0006226
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009268, 0.0009254
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011216, 0.0010974
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009115, 0.0008947
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019038, 0.0019381
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0065128, 0.0063621
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0063416, 0.0061629
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0042223, 0.0043298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019751, 0.0019893
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014725, 0.0014786
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006340, 0.0006290
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009318, 0.0009311
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0011131, 0.0011035
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0009071, 0.0009041
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0019106, 0.0019412
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0064587, 0.0064004
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0062760, 0.0062109
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0042498, 0.0042910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019611, 0.0019448
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014683, 0.0014494
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006193, 0.0006234
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009223, 0.0009346
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010794, 0.0010692
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0008904, 0.0009029
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018645, 0.0018925
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0062620, 0.0062103
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0060893, 0.0060536
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041281, 0.0041591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0153786, 0.0180746, 0.0153786, 0.0180746, -0.0019607, 0.0019483
1: -0.0017119, 0.0002095, -0.0017119, 0.0002095, -0.0014689, 0.0014514
2: 0.0036249, 0.0044952, 0.0036249, 0.0044952, -0.0006203, 0.0006233
3: 0.0013965, 0.0027650, 0.0013965, 0.0027650, -0.0009222, 0.0009350
4: -0.0046173, -0.0027327, -0.0046173, -0.0027327, -0.0010843, 0.0010678
5: -0.0002734, 0.0008871, -0.0002734, 0.0008871, -0.0008911, 0.0009035
6: -0.0048824, -0.0015696, -0.0048824, -0.0015696, -0.0018621, 0.0018975
7: -0.0227041, -0.0118535, -0.0227041, -0.0118535, -0.0062907, 0.0062013
8: 0.9747490, 0.9847534, 0.9747490, 0.9847534, -0.0061122, 0.0060463
9: 0.0000331, 0.0071689, 0.0000331, 0.0071689, -0.0041217, 0.0041780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
time: 0.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038606, upper bound: 0.0039904
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039739, upper bound: 0.0038666
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039739, upper bound: 0.0038666
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039741, upper bound: 0.0038665
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039741, upper bound: 0.0038665
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039739, upper bound: 0.0038666
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039739, upper bound: 0.0038666
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039741, upper bound: 0.0038665
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039741, upper bound: 0.0038665
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038665, upper bound: 0.0039741
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038665, upper bound: 0.0039741
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038666, upper bound: 0.0039739
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038666, upper bound: 0.0039739
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038665, upper bound: 0.0039741
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038665, upper bound: 0.0039741
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038666, upper bound: 0.0039739
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0038666, upper bound: 0.0039739
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 8, lower bound: -0.0039904, upper bound: 0.0038606

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.05 + 81.71 = 84.77 seconds
