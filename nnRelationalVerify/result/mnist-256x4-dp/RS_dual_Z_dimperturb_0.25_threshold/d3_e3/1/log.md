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
Threshold: 0.00206416


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0021752, 0.0021752)
1: (0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0003143, 0.0003143)
2: (0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0012026, 0.0012026)
3: (-0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0012438, 0.0012438)
4: (-0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0013465, 0.0013465)
5: (0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0012742, 0.0012742)
6: (-0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0050558, 0.0050558)
7: (0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0068855, 0.0068855)
8: (0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0048503, 0.0048503)
9: (-0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0044028, 0.0044028)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 1.56 = 3.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0026870, upper bound: 0.0026870

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024203, upper bound: 0.0025898
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025898, upper bound: 0.0024203
time: 0.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 8, lower bound: -0.0024203, upper bound: 0.0025898
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 8, lower bound: -0.0025898, upper bound: 0.0024203

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0021301, 0.0021644
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0003077, 0.0003127
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0011966, 0.0011777
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0012376, 0.0012180
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0013186, 0.0013398
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0012679, 0.0012478
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0050307, 0.0049510
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0067429, 0.0068513
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0047498, 0.0048262
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0043809, 0.0043116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022506, upper bound: 0.0023668
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022347, upper bound: 0.0023731
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0021752, 0.0021301
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0003143, 0.0003077
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0011777, 0.0012026
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0012180, 0.0012438
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0013465, 0.0013186
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0012478, 0.0012742
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0049510, 0.0050558
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0068855, 0.0067429
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0048503, 0.0047498
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0043116, 0.0044028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0022347
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023668, upper bound: 0.0022505
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 8, lower bound: -0.0022506, upper bound: 0.0023668
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 8, lower bound: -0.0022347, upper bound: 0.0023731
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0022347
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 8, lower bound: -0.0023668, upper bound: 0.0022505

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0020891, 0.0020964
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0003018, 0.0003029
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0011590, 0.0011550
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0011987, 0.0011946
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0012932, 0.0012977
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0012280, 0.0012238
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0048725, 0.0048556
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0066129, 0.0066359
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0046583, 0.0046745
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0042432, 0.0042285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019603, upper bound: 0.0021510
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020204, upper bound: 0.0020528
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0020621, 0.0021192
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0002979, 0.0003062
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0011716, 0.0011401
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0012118, 0.0011791
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0012765, 0.0013118
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0012414, 0.0012080
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0049256, 0.0047928
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0065274, 0.0067082
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0045981, 0.0047254
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0042894, 0.0041738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019391, upper bound: 0.0021577
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020069, upper bound: 0.0020664
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0021342, 0.0020621
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0003083, 0.0002979
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0011401, 0.0011799
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0011791, 0.0012203
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0013211, 0.0012765
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0012080, 0.0012502
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0047928, 0.0049604
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0067556, 0.0065274
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0047588, 0.0045981
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0041738, 0.0043197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020664, upper bound: 0.0020069
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021577, upper bound: 0.0019391
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0021072, 0.0020891
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0003044, 0.0003018
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0011550, 0.0011650
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0011946, 0.0012049
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0013044, 0.0012932
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0012238, 0.0012344
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0048556, 0.0048976
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0066701, 0.0066129
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0046986, 0.0046583
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0042285, 0.0042650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020528, upper bound: 0.0020204
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021510, upper bound: 0.0019603
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0019603, upper bound: 0.0021510
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0020204, upper bound: 0.0020528
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0019391, upper bound: 0.0021577
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0020069, upper bound: 0.0020664
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0020664, upper bound: 0.0020069
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0021577, upper bound: 0.0019391
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0020528, upper bound: 0.0020204
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 8, lower bound: -0.0021510, upper bound: 0.0019603

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0015469, 0.0015741
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0002235, 0.0002274
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0008703, 0.0008552
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0009001, 0.0008845
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0009576, 0.0009744
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0009221, 0.0009062
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0036587, 0.0035954
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0048966, 0.0049829
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0034493, 0.0035101
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0031862, 0.0031310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018519, upper bound: 0.0020043
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018519, upper bound: 0.0020043
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0015199, 0.0015976
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0002196, 0.0002308
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0008833, 0.0008403
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0009135, 0.0008691
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0009408, 0.0009889
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0009359, 0.0008903
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0037132, 0.0035326
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0048111, 0.0050571
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0033890, 0.0035623
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0032336, 0.0030764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018444, upper bound: 0.0020077
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018444, upper bound: 0.0020077
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0015376, 0.0015770
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0002221, 0.0002278
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0008719, 0.0008501
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0009017, 0.0008792
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0009518, 0.0009762
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0009238, 0.0009007
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0036653, 0.0035739
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0048673, 0.0049919
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0034286, 0.0035164
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0031919, 0.0031123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019062, upper bound: 0.0019267
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019062, upper bound: 0.0019267
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0015884, 0.0015376
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0002295, 0.0002221
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0008501, 0.0008782
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0008792, 0.0009083
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0009832, 0.0009518
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0009007, 0.0009305
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0035739, 0.0036919
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0050280, 0.0048673
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0035418, 0.0034286
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0031123, 0.0032150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019267, upper bound: 0.0019062
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019267, upper bound: 0.0019062
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0016083, 0.0015199
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0002324, 0.0002196
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0008403, 0.0008892
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0008691, 0.0009197
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0009956, 0.0009408
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0008903, 0.0009422
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0035326, 0.0037382
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0050911, 0.0048111
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0035863, 0.0033890
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0030764, 0.0032554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020077, upper bound: 0.0018442
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020077, upper bound: 0.0018444
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0062100, 0.0087816, 0.0062100, 0.0087816, -0.0015791, 0.0015469
1: 0.0022195, 0.0025910, 0.0022195, 0.0025910, -0.0002281, 0.0002235
2: 0.0095048, 0.0109265, 0.0095048, 0.0109265, -0.0008552, 0.0008731
3: -0.0048502, -0.0033797, -0.0048502, -0.0033797, -0.0008845, 0.0009030
4: -0.0003782, 0.0012137, -0.0003782, 0.0012137, -0.0009775, 0.0009576
5: 0.0029648, 0.0044712, 0.0029648, 0.0044712, -0.0009062, 0.0009251
6: -0.0105368, -0.0045598, -0.0105368, -0.0045598, -0.0035954, 0.0036703
7: 0.0036533, 0.0117935, 0.0036533, 0.0117935, -0.0049987, 0.0048966
8: 0.9917873, 0.9975215, 0.9917873, 0.9975215, -0.0035212, 0.0034493
9: -0.0136374, -0.0084324, -0.0136374, -0.0084324, -0.0031310, 0.0031963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020043, upper bound: 0.0018519
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020043, upper bound: 0.0018519
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0018519, upper bound: 0.0020043
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0018519, upper bound: 0.0020043
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0018444, upper bound: 0.0020077
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0018444, upper bound: 0.0020077
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0019062, upper bound: 0.0019267
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0019062, upper bound: 0.0019267
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0019267, upper bound: 0.0019062
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0019267, upper bound: 0.0019062
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0020077, upper bound: 0.0018442
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0020077, upper bound: 0.0018444
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0020043, upper bound: 0.0018519
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 8, lower bound: -0.0020043, upper bound: 0.0018519

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.22 + 55.32 = 58.53 seconds
