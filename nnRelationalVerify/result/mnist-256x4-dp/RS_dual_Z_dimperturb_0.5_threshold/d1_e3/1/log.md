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
Threshold: 0.000371


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000161, 0.0000161)
1: (-0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0006046, 0.0006046)
2: (0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0007256, 0.0007256)
3: (0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0053518, 0.0053518)
4: (-0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0004070, 0.0004070)
5: (0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0004114, 0.0004114)
6: (0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0002001, 0.0002001)
7: (-0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0013870, 0.0013870)
8: (0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0011003, 0.0011003)
9: (0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0019791, 0.0019791)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.65 = 2.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0005119, upper bound: 0.0005119

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004132, upper bound: 0.0004132
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004132, upper bound: 0.0004132
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 2, lower bound: -0.0004132, upper bound: 0.0004132
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 2, lower bound: -0.0004132, upper bound: 0.0004132

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000161, 0.0000161
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0006046, 0.0006046
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0007255, 0.0007255
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0053511, 0.0053511
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0004070, 0.0004070
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0004113, 0.0004113
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0002001, 0.0002001
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0013868, 0.0013868
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0011002, 0.0011002
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0019788, 0.0019788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003939, upper bound: 0.0003940
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003940, upper bound: 0.0003939
time: 0.84 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000161, 0.0000161
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0006046, 0.0006046
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0007256, 0.0007255
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0053518, 0.0053511
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0004070, 0.0004070
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0004113, 0.0004114
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0002001, 0.0002001
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0013868, 0.0013870
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0011002, 0.0011003
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0019788, 0.0019791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003939, upper bound: 0.0003940
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003940, upper bound: 0.0003939
time: 0.85 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 2, lower bound: -0.0003939, upper bound: 0.0003940
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 2, lower bound: -0.0003940, upper bound: 0.0003939
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 2, lower bound: -0.0003939, upper bound: 0.0003940
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 2, lower bound: -0.0003940, upper bound: 0.0003939

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000154, 0.0000154
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005782, 0.0005780
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006938, 0.0006936
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0051176, 0.0051157
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003891, 0.0003892
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003932, 0.0003934
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001913, 0.0001913
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0013258, 0.0013263
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0010518, 0.0010522
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0018918, 0.0018925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003668, upper bound: 0.0003813
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003812, upper bound: 0.0003668
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000154, 0.0000154
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005779, 0.0005782
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006935, 0.0006938
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0051153, 0.0051176
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003892, 0.0003890
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003934, 0.0003932
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001913, 0.0001913
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0013263, 0.0013257
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0010522, 0.0010517
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0018925, 0.0018916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003668, upper bound: 0.0003812
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003813, upper bound: 0.0003668
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000154, 0.0000154
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005782, 0.0005779
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006939, 0.0006935
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0051182, 0.0051153
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003890, 0.0003893
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003932, 0.0003934
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001914, 0.0001913
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0013257, 0.0013264
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0010517, 0.0010523
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0018916, 0.0018927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003668, upper bound: 0.0003813
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003812, upper bound: 0.0003668
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000154, 0.0000154
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005780, 0.0005782
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006936, 0.0006938
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0051160, 0.0051176
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003892, 0.0003891
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003934, 0.0003933
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001913, 0.0001913
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0013263, 0.0013258
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0010522, 0.0010519
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0018925, 0.0018919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003668, upper bound: 0.0003812
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003813, upper bound: 0.0003668
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 2, lower bound: -0.0003668, upper bound: 0.0003813
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 2, lower bound: -0.0003812, upper bound: 0.0003668
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 2, lower bound: -0.0003668, upper bound: 0.0003812
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 2, lower bound: -0.0003813, upper bound: 0.0003668
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 2, lower bound: -0.0003668, upper bound: 0.0003813
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 2, lower bound: -0.0003812, upper bound: 0.0003668
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 2, lower bound: -0.0003668, upper bound: 0.0003812
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 2, lower bound: -0.0003813, upper bound: 0.0003668

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000139, 0.0000142
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005220, 0.0005326
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006264, 0.0006392
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0046202, 0.0047145
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003586, 0.0003514
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003624, 0.0003551
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001727, 0.0001763
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0012218, 0.0011974
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009693, 0.0009499
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0017434, 0.0017086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003730
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003729
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000142, 0.0000139
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005321, 0.0005218
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006386, 0.0006261
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0047101, 0.0046183
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003513, 0.0003582
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003550, 0.0003621
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001761, 0.0001727
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011969, 0.0012207
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009495, 0.0009684
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0017079, 0.0017418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003586
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003585
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000139, 0.0000142
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005217, 0.0005322
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006261, 0.0006387
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0046179, 0.0047107
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003583, 0.0003512
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003621, 0.0003550
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001727, 0.0001761
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0012208, 0.0011968
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009685, 0.0009495
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0017420, 0.0017077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003585, upper bound: 0.0003729
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003729
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000142, 0.0000139
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005325, 0.0005220
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006391, 0.0006264
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0047136, 0.0046202
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003514, 0.0003585
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003551, 0.0003623
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001762, 0.0001727
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011974, 0.0012216
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009499, 0.0009691
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0017086, 0.0017431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003586
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003730, upper bound: 0.0003586
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000139, 0.0000142
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005221, 0.0005325
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006265, 0.0006391
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0046209, 0.0047136
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003585, 0.0003514
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003623, 0.0003552
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001728, 0.0001762
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0012216, 0.0011975
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009691, 0.0009501
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0017431, 0.0017088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003730
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003729
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000142, 0.0000139
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005322, 0.0005217
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006387, 0.0006261
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0047107, 0.0046179
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003512, 0.0003583
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003550, 0.0003621
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001761, 0.0001727
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011968, 0.0012208
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009495, 0.0009685
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0017077, 0.0017420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003586
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003585
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000139, 0.0000142
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005218, 0.0005321
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006262, 0.0006386
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0046186, 0.0047101
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003582, 0.0003513
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003621, 0.0003550
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001727, 0.0001761
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0012207, 0.0011970
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009684, 0.0009496
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0017418, 0.0017080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003585, upper bound: 0.0003729
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003729
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000142, 0.0000139
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005326, 0.0005220
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006391, 0.0006264
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0047143, 0.0046202
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003514, 0.0003585
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003551, 0.0003624
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001763, 0.0001727
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011974, 0.0012217
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009499, 0.0009693
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0017086, 0.0017433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003586
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003730, upper bound: 0.0003586
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003730
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003729
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003586
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003585
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003585, upper bound: 0.0003729
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003729
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003586
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003730, upper bound: 0.0003586
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003730
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003729
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003586
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003585
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003585, upper bound: 0.0003729
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003586, upper bound: 0.0003729
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003729, upper bound: 0.0003586
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 2, lower bound: -0.0003730, upper bound: 0.0003586

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000133, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0004975, 0.0005111
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0005970, 0.0006133
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044033, 0.0045238
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003441, 0.0003349
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003477, 0.0003385
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001646, 0.0001691
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011724, 0.0011411
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009301, 0.0009053
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016729, 0.0016283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003691
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003727
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000134, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005004, 0.0005076
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006005, 0.0006092
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044295, 0.0044934
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003417, 0.0003369
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003454, 0.0003405
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001656, 0.0001680
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011645, 0.0011480
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009239, 0.0009107
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016616, 0.0016380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003688
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000135, 0.0000134
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005072, 0.0005002
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006087, 0.0006003
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044898, 0.0044276
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003367, 0.0003415
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003403, 0.0003451
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001679, 0.0001655
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011475, 0.0011636
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009103, 0.0009231
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016373, 0.0016603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003683, upper bound: 0.0003583
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000133
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005106, 0.0004968
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006127, 0.0005962
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045194, 0.0043973
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003344, 0.0003437
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003380, 0.0003474
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001690, 0.0001644
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011396, 0.0011712
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009041, 0.0009292
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016261, 0.0016713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003684, upper bound: 0.0003582
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000133, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0004967, 0.0005107
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0005961, 0.0006128
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0043968, 0.0045200
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003438, 0.0003344
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003474, 0.0003380
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001644, 0.0001690
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011714, 0.0011395
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009293, 0.0009040
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016715, 0.0016259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003582, upper bound: 0.0003684
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000134, 0.0000135
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005002, 0.0005073
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006002, 0.0006088
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044273, 0.0044905
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003415, 0.0003367
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003452, 0.0003403
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001655, 0.0001679
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011637, 0.0011474
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009233, 0.0009103
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016606, 0.0016372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003683
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000134
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005075, 0.0005004
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006091, 0.0006005
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044923, 0.0044295
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003369, 0.0003417
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003405, 0.0003453
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001680, 0.0001656
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011480, 0.0011642
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009107, 0.0009236
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016380, 0.0016612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003688, upper bound: 0.0003583
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000133
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005110, 0.0004974
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006132, 0.0005970
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045229, 0.0044030
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003349, 0.0003440
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003385, 0.0003477
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001691, 0.0001646
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011411, 0.0011721
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009053, 0.0009299
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016282, 0.0016726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003727, upper bound: 0.0003551
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003691, upper bound: 0.0003583
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000133, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0004976, 0.0005110
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0005971, 0.0006132
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044040, 0.0045229
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003440, 0.0003349
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003477, 0.0003385
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001647, 0.0001691
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011721, 0.0011413
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009299, 0.0009055
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016726, 0.0016286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003691
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003727
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000134, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005005, 0.0005075
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006006, 0.0006091
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044302, 0.0044923
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003417, 0.0003369
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003453, 0.0003405
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001656, 0.0001680
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011642, 0.0011481
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009236, 0.0009109
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016612, 0.0016383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003688
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000135, 0.0000134
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005073, 0.0005002
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006088, 0.0006002
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044905, 0.0044273
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003367, 0.0003415
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003403, 0.0003452
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001679, 0.0001655
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011474, 0.0011638
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009103, 0.0009233
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016372, 0.0016606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003683, upper bound: 0.0003583
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000133
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005107, 0.0004967
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006128, 0.0005961
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045201, 0.0043968
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003344, 0.0003438
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003380, 0.0003474
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001690, 0.0001644
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011395, 0.0011714
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009040, 0.0009293
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016259, 0.0016715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003684, upper bound: 0.0003582
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000133, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0004968, 0.0005106
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0005962, 0.0006127
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0043975, 0.0045194
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003437, 0.0003345
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003474, 0.0003380
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001644, 0.0001690
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011712, 0.0011397
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009292, 0.0009042
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016713, 0.0016262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003582, upper bound: 0.0003684
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000134, 0.0000135
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005003, 0.0005072
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006003, 0.0006087
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044280, 0.0044898
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003415, 0.0003368
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003451, 0.0003404
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001656, 0.0001679
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011636, 0.0011475
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009231, 0.0009104
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016603, 0.0016375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003683
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000134
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005076, 0.0005004
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006092, 0.0006005
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044930, 0.0044295
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003369, 0.0003417
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003405, 0.0003454
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001680, 0.0001656
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011480, 0.0011644
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009107, 0.0009238
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016380, 0.0016615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003688, upper bound: 0.0003583
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000133
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005111, 0.0004975
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006133, 0.0005970
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045236, 0.0044033
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003349, 0.0003440
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003385, 0.0003477
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001691, 0.0001646
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011411, 0.0011723
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009053, 0.0009301
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016283, 0.0016728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003727, upper bound: 0.0003551
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003691, upper bound: 0.0003583
time: 0.72 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003691
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003727
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003688
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003683, upper bound: 0.0003583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003684, upper bound: 0.0003582
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003582, upper bound: 0.0003684
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003683
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003688, upper bound: 0.0003583
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003727, upper bound: 0.0003551
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003691, upper bound: 0.0003583
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003691
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003727
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003688
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003683, upper bound: 0.0003583
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003684, upper bound: 0.0003582
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003582, upper bound: 0.0003684
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003583, upper bound: 0.0003683
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003551, upper bound: 0.0003726
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003551
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003688, upper bound: 0.0003583
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003727, upper bound: 0.0003551
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.69
Output dim: 2, lower bound: -0.0003691, upper bound: 0.0003583

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000133, 0.0000137
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0004985, 0.0005134
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0005982, 0.0006161
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044125, 0.0045442
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003456, 0.0003356
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003493, 0.0003392
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001650, 0.0001699
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011777, 0.0011435
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009343, 0.0009072
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016804, 0.0016317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002761, upper bound: 0.0002890
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002761, upper bound: 0.0002890
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000134, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005016, 0.0005100
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006020, 0.0006120
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044401, 0.0045138
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003433, 0.0003377
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003470, 0.0003413
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001660, 0.0001688
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011698, 0.0011507
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009281, 0.0009129
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016692, 0.0016419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002760, upper bound: 0.0002892
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002760, upper bound: 0.0002892
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000134
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005096, 0.0005017
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006115, 0.0006020
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045102, 0.0044403
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003377, 0.0003430
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003413, 0.0003467
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001686, 0.0001660
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011507, 0.0011689
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009130, 0.0009273
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016420, 0.0016679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002764
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002764
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000137, 0.0000133
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005129, 0.0004983
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006155, 0.0005979
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045398, 0.0044102
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003354, 0.0003453
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003390, 0.0003490
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001697, 0.0001649
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011430, 0.0011765
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009068, 0.0009334
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016309, 0.0016788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002764
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002764
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000133, 0.0000137
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0004982, 0.0005130
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0005979, 0.0006156
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044098, 0.0045405
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003453, 0.0003354
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003490, 0.0003390
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001649, 0.0001698
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011767, 0.0011428
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009335, 0.0009067
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016791, 0.0016307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002890
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002890
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000134, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005016, 0.0005096
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006019, 0.0006116
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044398, 0.0045109
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003431, 0.0003377
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003467, 0.0003413
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001660, 0.0001687
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011690, 0.0011506
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009275, 0.0009128
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016681, 0.0016418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002892
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002892
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000134
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005098, 0.0005016
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006118, 0.0006020
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045127, 0.0044400
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003377, 0.0003432
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003413, 0.0003469
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001687, 0.0001660
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011507, 0.0011695
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009129, 0.0009278
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016419, 0.0016688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002760
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002760
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000137, 0.0000133
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005133, 0.0004985
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006160, 0.0005982
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045433, 0.0044123
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003356, 0.0003455
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003392, 0.0003492
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001699, 0.0001650
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011435, 0.0011774
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009072, 0.0009341
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016317, 0.0016801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002761
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002761
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000133, 0.0000137
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0004986, 0.0005133
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0005983, 0.0006160
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044132, 0.0045433
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003455, 0.0003356
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003492, 0.0003392
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001650, 0.0001699
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011774, 0.0011437
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009341, 0.0009074
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016801, 0.0016320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002761, upper bound: 0.0002890
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002761, upper bound: 0.0002890
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000134, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005017, 0.0005098
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006021, 0.0006118
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044408, 0.0045127
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003432, 0.0003377
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003469, 0.0003414
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001660, 0.0001687
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011695, 0.0011509
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009278, 0.0009130
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016688, 0.0016422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002760, upper bound: 0.0002892
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002760, upper bound: 0.0002892
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000134
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005096, 0.0005016
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006116, 0.0006019
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045109, 0.0044398
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003377, 0.0003431
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003413, 0.0003467
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001687, 0.0001660
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011506, 0.0011690
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009128, 0.0009275
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016418, 0.0016681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002764
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002764
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000137, 0.0000133
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005130, 0.0004982
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006156, 0.0005979
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045405, 0.0044098
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003354, 0.0003453
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003390, 0.0003490
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001698, 0.0001649
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011428, 0.0011767
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009067, 0.0009335
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016307, 0.0016791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002764
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002764
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000133, 0.0000137
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0004983, 0.0005129
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0005980, 0.0006155
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044105, 0.0045398
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003453, 0.0003354
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003490, 0.0003390
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001649, 0.0001697
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011765, 0.0011430
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009334, 0.0009068
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016788, 0.0016310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002890
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002890
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000134, 0.0000136
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005017, 0.0005096
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006020, 0.0006115
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0044405, 0.0045102
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003430, 0.0003377
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003467, 0.0003413
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001660, 0.0001686
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011689, 0.0011508
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009273, 0.0009130
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016679, 0.0016421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002892
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002892
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000136, 0.0000134
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005099, 0.0005016
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006119, 0.0006020
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045134, 0.0044401
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003377, 0.0003433
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003413, 0.0003469
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001687, 0.0001660
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011507, 0.0011697
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009129, 0.0009280
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016419, 0.0016691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002760
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002760
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041060, -0.0040753, -0.0041060, -0.0040753, -0.0000137, 0.0000133
1: -0.0064269, -0.0052769, -0.0064269, -0.0052769, -0.0005134, 0.0004985
2: 0.9687509, 0.9701309, 0.9687509, 0.9701309, -0.0006161, 0.0005982
3: 0.0158177, 0.0259965, 0.0158177, 0.0259965, -0.0045440, 0.0044125
4: -0.0026702, -0.0018961, -0.0026702, -0.0018961, -0.0003356, 0.0003456
5: 0.0145716, 0.0153540, 0.0145716, 0.0153540, -0.0003392, 0.0003493
6: 0.0044386, 0.0048192, 0.0044386, 0.0048192, -0.0001699, 0.0001650
7: -0.0145155, -0.0118775, -0.0145155, -0.0118775, -0.0011435, 0.0011776
8: 0.0052133, 0.0073061, 0.0052133, 0.0073061, -0.0009072, 0.0009343
9: 0.0071011, 0.0108653, 0.0071011, 0.0108653, -0.0016317, 0.0016804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002761
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002761
time: 0.68 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002761, upper bound: 0.0002890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002761, upper bound: 0.0002890
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002760, upper bound: 0.0002892
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002760, upper bound: 0.0002892
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002764
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002764
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002764
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002764
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002890
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002890
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002892
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002892
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002760
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002760
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002761
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002761
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002761, upper bound: 0.0002890
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002761, upper bound: 0.0002890
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002760, upper bound: 0.0002892
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002760, upper bound: 0.0002892
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002764
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002764
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002764
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002764
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002890
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002892
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002764, upper bound: 0.0002892
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002760
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002892, upper bound: 0.0002760
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002761
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 2, lower bound: -0.0002890, upper bound: 0.0002761

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.94 + 129.90 = 132.83 seconds
