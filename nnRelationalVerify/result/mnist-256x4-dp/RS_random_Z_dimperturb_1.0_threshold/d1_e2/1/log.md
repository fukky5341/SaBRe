## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00158454


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741)
1: (0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977)
2: (0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0107039, 0.0107039)
3: (-0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259)
4: (0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0016757, 0.0016757)
5: (-0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670)
6: (-0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753)
7: (-0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175)
8: (-0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897)
9: (1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.08 + 2.71 = 3.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0017606, upper bound: 0.0017606

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017480, upper bound: 0.0017480
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017480, upper bound: 0.0017480
time: 1.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.71
Output dim: 9, lower bound: -0.0017480, upper bound: 0.0017480
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.71
Output dim: 9, lower bound: -0.0017480, upper bound: 0.0017480

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0106078, 0.0106405
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0016486, 0.0016256
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0013466, upper bound: 0.0013467
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0013466, upper bound: 0.0013467
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0106405, 0.0106078
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0016256, 0.0016486
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017470, upper bound: 0.0017470
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017470, upper bound: 0.0017470
time: 1.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.39 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.39
Output dim: 9, lower bound: -0.0013466, upper bound: 0.0013467
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.39
Output dim: 9, lower bound: -0.0013466, upper bound: 0.0013467
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.39
Output dim: 9, lower bound: -0.0017470, upper bound: 0.0017470
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.39
Output dim: 9, lower bound: -0.0017470, upper bound: 0.0017470

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0105270, 0.0104408
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015455, 0.0015955
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017407, upper bound: 0.0017341
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017341, upper bound: 0.0017407
time: 2.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0104735, 0.0106078
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0016256, 0.0015685
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017293, upper bound: 0.0017245
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017245, upper bound: 0.0017292
time: 1.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.75 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 9, lower bound: -0.0017407, upper bound: 0.0017341
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 9, lower bound: -0.0017341, upper bound: 0.0017407
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 9, lower bound: -0.0017293, upper bound: 0.0017245
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 9, lower bound: -0.0017245, upper bound: 0.0017292

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0103508, 0.0103166
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015096, 0.0015584
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017108, upper bound: 0.0017108
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017108, upper bound: 0.0017109
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0103907, 0.0102645
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015084, 0.0015580
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017108, upper bound: 0.0017171
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017108, upper bound: 0.0017171
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100638, 0.0103473
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015426, 0.0014331
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017028, upper bound: 0.0016989
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017028, upper bound: 0.0016989
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0102065, 0.0102011
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014958, 0.0014789
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017198, upper bound: 0.0017244
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017198, upper bound: 0.0017244
time: 1.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.84 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 9, lower bound: -0.0017108, upper bound: 0.0017108
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 9, lower bound: -0.0017108, upper bound: 0.0017109
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 9, lower bound: -0.0017108, upper bound: 0.0017171
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 9, lower bound: -0.0017108, upper bound: 0.0017171
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 9, lower bound: -0.0017028, upper bound: 0.0016989
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 9, lower bound: -0.0017028, upper bound: 0.0016989
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 9, lower bound: -0.0017198, upper bound: 0.0017244
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 9, lower bound: -0.0017198, upper bound: 0.0017244

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101608, 0.0101524
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014522, 0.0014806
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017124, upper bound: 0.0017061
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017061, upper bound: 0.0017061
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101897, 0.0101266
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014319, 0.0014971
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016886, upper bound: 0.0016893
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016886, upper bound: 0.0016932
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0102007, 0.0101126
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014543, 0.0014803
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017042, upper bound: 0.0017096
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017042, upper bound: 0.0017096
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0102198, 0.0100745
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014307, 0.0014947
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016924, upper bound: 0.0016980
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016924, upper bound: 0.0016980
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100392, 0.0103373
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015386, 0.0014187
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016866, upper bound: 0.0016866
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016866, upper bound: 0.0016920
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100638, 0.0103227
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015282, 0.0014331
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016898, upper bound: 0.0016899
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016898, upper bound: 0.0016899
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101973, 0.0101931
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014850, 0.0014666
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016407, upper bound: 0.0016424
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016407, upper bound: 0.0016424
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101993, 0.0101912
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014818, 0.0014685
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017073, upper bound: 0.0017116
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017073, upper bound: 0.0017179
time: 1.76 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.42 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0017124, upper bound: 0.0017061
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0017061, upper bound: 0.0017061
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016886, upper bound: 0.0016893
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016886, upper bound: 0.0016932
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0017042, upper bound: 0.0017096
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0017042, upper bound: 0.0017096
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016924, upper bound: 0.0016980
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016924, upper bound: 0.0016980
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016866, upper bound: 0.0016866
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016866, upper bound: 0.0016920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016898, upper bound: 0.0016899
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016898, upper bound: 0.0016899
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016407, upper bound: 0.0016424
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0016407, upper bound: 0.0016424
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0017073, upper bound: 0.0017116
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 9, lower bound: -0.0017073, upper bound: 0.0017179

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101522, 0.0101459
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014453, 0.0014696
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016842, upper bound: 0.0016843
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016842, upper bound: 0.0016884
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101541, 0.0101438
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014411, 0.0014736
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016202, upper bound: 0.0016202
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016202, upper bound: 0.0016202
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097626, 0.0098388
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013357, 0.0013527
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016699, upper bound: 0.0016707
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016699, upper bound: 0.0016707
time: 2.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098943, 0.0096995
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012875, 0.0013945
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016026, upper bound: 0.0016037
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016026, upper bound: 0.0016037
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101783, 0.0100854
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014254, 0.0014669
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016190, upper bound: 0.0016240
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016190, upper bound: 0.0016240
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101735, 0.0101126
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014543, 0.0014514
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016820, upper bound: 0.0016868
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016820, upper bound: 0.0016917
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101914, 0.0100466
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014228, 0.0014886
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016003, upper bound: 0.0016045
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016003, upper bound: 0.0016045
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0102198, 0.0100462
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014307, 0.0014868
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016003, upper bound: 0.0016045
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016003, upper bound: 0.0016045
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098487, 0.0101879
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015019, 0.0013817
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016786, upper bound: 0.0016787
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016786, upper bound: 0.0016787
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098996, 0.0101433
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015001, 0.0013828
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016820, upper bound: 0.0016872
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016820, upper bound: 0.0016872
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100414, 0.0102958
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015031, 0.0014210
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015659, upper bound: 0.0015658
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015658, upper bound: 0.0015658
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100382, 0.0103227
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015282, 0.0014106
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016850, upper bound: 0.0016851
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016896, upper bound: 0.0016851
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101603, 0.0101240
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014316, 0.0014436
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016140, upper bound: 0.0016154
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016140, upper bound: 0.0016154
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101292, 0.0101931
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014850, 0.0014135
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016289, upper bound: 0.0016308
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016289, upper bound: 0.0016366
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100099, 0.0100505
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014465, 0.0014334
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016896, upper bound: 0.0016936
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016896, upper bound: 0.0016936
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100530, 0.0099984
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014453, 0.0014357
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016289, upper bound: 0.0016366
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016289, upper bound: 0.0016366
time: 1.48 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.09 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016842, upper bound: 0.0016843
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016842, upper bound: 0.0016884
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016202, upper bound: 0.0016202
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016202, upper bound: 0.0016202
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016699, upper bound: 0.0016707
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016699, upper bound: 0.0016707
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016026, upper bound: 0.0016037
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016026, upper bound: 0.0016037
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016190, upper bound: 0.0016240
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016190, upper bound: 0.0016240
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016820, upper bound: 0.0016868
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016820, upper bound: 0.0016917
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016003, upper bound: 0.0016045
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016003, upper bound: 0.0016045
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016003, upper bound: 0.0016045
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016003, upper bound: 0.0016045
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016786, upper bound: 0.0016787
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016786, upper bound: 0.0016787
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016820, upper bound: 0.0016872
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016820, upper bound: 0.0016872
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0015659, upper bound: 0.0015658
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0015658, upper bound: 0.0015658
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016850, upper bound: 0.0016851
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016896, upper bound: 0.0016851
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016140, upper bound: 0.0016154
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016140, upper bound: 0.0016154
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016289, upper bound: 0.0016308
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016289, upper bound: 0.0016366
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016896, upper bound: 0.0016936
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016896, upper bound: 0.0016936
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016289, upper bound: 0.0016366
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 9, lower bound: -0.0016289, upper bound: 0.0016366

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097253, 0.0098615
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013523, 0.0013272
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016867, upper bound: 0.0016776
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016775, upper bound: 0.0016777
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098541, 0.0097190
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013030, 0.0013686
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016693
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101114, 0.0100744
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052208, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013903, 0.0014469
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015955
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015955
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100847, 0.0101438
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014411, 0.0014228
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016141
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016141
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097340, 0.0098171
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013279, 0.0013466
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016662
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016662
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097626, 0.0098102
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013357, 0.0013448
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016647, upper bound: 0.0016654
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016647, upper bound: 0.0016654
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098580, 0.0096281
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052210, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012333, 0.0013705
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015474, upper bound: 0.0015449
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015474, upper bound: 0.0015449
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098229, 0.0096995
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012875, 0.0013403
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016029, upper bound: 0.0015987
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016029, upper bound: 0.0015987
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101376, 0.0100162
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013736, 0.0014377
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015992
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0016006
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101091, 0.0100854
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014254, 0.0014151
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015508, upper bound: 0.0015534
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015508, upper bound: 0.0015534
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097471, 0.0098331
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013577, 0.0013079
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015992
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015992
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098732, 0.0096854
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013099, 0.0013498
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0016006
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0016006
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101533, 0.0099772
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052247, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013712, 0.0014646
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101220, 0.0100466
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014228, 0.0014370
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101816, 0.0099768
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013792, 0.0014632
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015961, upper bound: 0.0016000
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015961, upper bound: 0.0016000
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101504, 0.0100462
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014307, 0.0014352
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098255, 0.0101611
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014746, 0.0013668
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015568, upper bound: 0.0015568
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015568, upper bound: 0.0015568
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098223, 0.0101879
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015019, 0.0013564
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016554, upper bound: 0.0016556
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016554, upper bound: 0.0016560
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098908, 0.0101359
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014902, 0.0013722
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015723, upper bound: 0.0015751
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015717, upper bound: 0.0015751
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098926, 0.0101339
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014880, 0.0013751
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016556, upper bound: 0.0016598
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016556, upper bound: 0.0016598
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100289, 0.0103147
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015165, 0.0013987
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016739, upper bound: 0.0016740
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016739, upper bound: 0.0016784
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100303, 0.0103127
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015140, 0.0014012
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016739, upper bound: 0.0016740
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016739, upper bound: 0.0016784
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101302, 0.0100945
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014244, 0.0014365
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015810, upper bound: 0.0015819
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015810, upper bound: 0.0015815
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101603, 0.0100939
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014316, 0.0014366
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015434, upper bound: 0.0015436
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015434, upper bound: 0.0015436
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099373, 0.0100526
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014501, 0.0013769
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016214, upper bound: 0.0016234
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016214, upper bound: 0.0016234
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099804, 0.0100004
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014485, 0.0013786
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015977, upper bound: 0.0016039
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015977, upper bound: 0.0016033
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099815, 0.0100276
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014384, 0.0014261
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100099, 0.0100221
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014465, 0.0014253
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100111, 0.0099273
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013913, 0.0014118
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015717, upper bound: 0.0015756
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015717, upper bound: 0.0015756
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099824, 0.0099984
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014453, 0.0013812
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015977, upper bound: 0.0016039
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015977, upper bound: 0.0016033
time: 1.39 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.89 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016867, upper bound: 0.0016776
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016775, upper bound: 0.0016777
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016693
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015955
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015955
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016141
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016141
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016662
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016662
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016647, upper bound: 0.0016654
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016647, upper bound: 0.0016654
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015474, upper bound: 0.0015449
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015474, upper bound: 0.0015449
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016029, upper bound: 0.0015987
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016029, upper bound: 0.0015987
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015992
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0016006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015508, upper bound: 0.0015534
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015508, upper bound: 0.0015534
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015992
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015992
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0016006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0016006
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015961, upper bound: 0.0016000
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015961, upper bound: 0.0016000
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015955, upper bound: 0.0015996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015568, upper bound: 0.0015568
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015568, upper bound: 0.0015568
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016554, upper bound: 0.0016556
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016554, upper bound: 0.0016560
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015723, upper bound: 0.0015751
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015717, upper bound: 0.0015751
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016556, upper bound: 0.0016598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016556, upper bound: 0.0016598
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016739, upper bound: 0.0016740
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016739, upper bound: 0.0016784
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016739, upper bound: 0.0016740
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016739, upper bound: 0.0016784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015810, upper bound: 0.0015819
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015810, upper bound: 0.0015815
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015434, upper bound: 0.0015436
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015434, upper bound: 0.0015436
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016214, upper bound: 0.0016234
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016214, upper bound: 0.0016234
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015977, upper bound: 0.0016039
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015977, upper bound: 0.0016033
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0016655, upper bound: 0.0016694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015717, upper bound: 0.0015756
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015717, upper bound: 0.0015756
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015977, upper bound: 0.0016039
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.89
Output dim: 9, lower bound: -0.0015977, upper bound: 0.0016033

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097091, 0.0098350
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013250, 0.0013155
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016604
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016604
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0096987, 0.0098615
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013523, 0.0012999
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015906
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015906
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098255, 0.0096959
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012950, 0.0013623
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015729
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015729
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098541, 0.0096904
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013030, 0.0013605
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016312, upper bound: 0.0016347
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016312, upper bound: 0.0016347
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100831, 0.0100516
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052004, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013824, 0.0014408
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015913
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015913
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101114, 0.0100461
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052208, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013903, 0.0014389
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015913
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015913
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100685, 0.0101167
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014125, 0.0014094
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015906
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015916
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100579, 0.0101438
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052256
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014411, 0.0013940
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015457, upper bound: 0.0015457
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015457, upper bound: 0.0015457
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097256, 0.0098106
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052234, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013219, 0.0013374
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016312, upper bound: 0.0016318
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016312, upper bound: 0.0016318
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097279, 0.0098087
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052220, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013187, 0.0013410
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016608
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016608
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097445, 0.0097837
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013077, 0.0013310
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097360, 0.0098102
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013357, 0.0013173
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098144, 0.0096932
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012821, 0.0013324
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015942, upper bound: 0.0015916
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015916
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098171, 0.0096911
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012785, 0.0013361
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015421, upper bound: 0.0015397
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015421, upper bound: 0.0015397
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097091, 0.0097354
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052193, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012760, 0.0012923
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015752
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015752
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098347, 0.0095877
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052190, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012282, 0.0013371
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015762
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015762
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097036, 0.0097618
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013035, 0.0012780
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015730, upper bound: 0.0015752
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015730, upper bound: 0.0015752
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0096758, 0.0098331
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052179
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013577, 0.0012542
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015251, upper bound: 0.0015269
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015251, upper bound: 0.0015269
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098305, 0.0096141
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012557, 0.0013252
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015762
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015762
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098020, 0.0096854
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052184
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013099, 0.0012962
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101447, 0.0099707
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052082, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013641, 0.0014542
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015950
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015950
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101466, 0.0099686
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052068, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013608, 0.0014573
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015263, upper bound: 0.0015280
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015263, upper bound: 0.0015280
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101134, 0.0100401
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014151, 0.0014266
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015755
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
time: 3.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101157, 0.0100380
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014118, 0.0014303
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015730, upper bound: 0.0015755
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101596, 0.0099498
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052179, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013499, 0.0014457
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015749
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101547, 0.0099768
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013792, 0.0014344
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0015211
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0015211
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101418, 0.0100395
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052221
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014228, 0.0014248
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015263, upper bound: 0.0015280
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015263, upper bound: 0.0015280
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101440, 0.0100376
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052254
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014196, 0.0014288
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015755
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0096281, 0.0100234
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014362, 0.0012665
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0096670, 0.0099946
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014139, 0.0012892
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098643, 0.0101083
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014799, 0.0013674
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015353, upper bound: 0.0015374
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015355, upper bound: 0.0015374
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098926, 0.0101057
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014880, 0.0013670
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098387, 0.0101663
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014834, 0.0013605
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016514
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098896, 0.0101214
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014800, 0.0013616
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098402, 0.0101646
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014801, 0.0013630
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016514
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098914, 0.0101193
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014775, 0.0013645
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099139, 0.0100258
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014223, 0.0013611
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015940, upper bound: 0.0015917
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015942, upper bound: 0.0015916
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099112, 0.0100526
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014501, 0.0013517
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015991, upper bound: 0.0016004
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015991, upper bound: 0.0016004
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097861, 0.0098463
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052156
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013875, 0.0012946
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015764
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015763
time: 2.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098150, 0.0098076
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052252
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013635, 0.0013150
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015951
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015951
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097878, 0.0098606
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013739, 0.0013406
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016642
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016642
time: 1.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098276, 0.0098346
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013536, 0.0013632
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015728
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015728
time: 2.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098164, 0.0098550
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013818, 0.0013398
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015728
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015728
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098562, 0.0098291
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013614, 0.0013625
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016643
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016643
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097881, 0.0098437
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052172
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013838, 0.0012971
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098171, 0.0098057
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013602, 0.0013180
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
time: 1.89 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.44 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016604
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016604
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015906
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015906
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015729
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015729
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016312, upper bound: 0.0016347
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016312, upper bound: 0.0016347
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015913
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015913
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015913
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015913
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015906
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015916
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015457, upper bound: 0.0015457
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015457, upper bound: 0.0015457
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016312, upper bound: 0.0016318
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016312, upper bound: 0.0016318
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016608
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016608
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015942, upper bound: 0.0015916
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015916
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015421, upper bound: 0.0015397
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015421, upper bound: 0.0015397
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015752
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015752
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015762
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015762
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015730, upper bound: 0.0015752
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015730, upper bound: 0.0015752
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015251, upper bound: 0.0015269
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015251, upper bound: 0.0015269
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015762
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015762
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015950
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015950
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015263, upper bound: 0.0015280
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015263, upper bound: 0.0015280
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015730, upper bound: 0.0015755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015749
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0015211
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0015211
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015263, upper bound: 0.0015280
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015263, upper bound: 0.0015280
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015353, upper bound: 0.0015374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015355, upper bound: 0.0015374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016514
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016514
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015940, upper bound: 0.0015917
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015942, upper bound: 0.0015916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015991, upper bound: 0.0016004
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015991, upper bound: 0.0016004
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015764
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015763
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015951
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016642
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016642
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016643
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016643
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.44
Output dim: 9, lower bound: -0.0015722, upper bound: 0.0015759

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0096805, 0.0098134
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052063, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013176, 0.0013106
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016254
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016254
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097091, 0.0098063
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013250, 0.0013081
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0096541, 0.0097901
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052150, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012993, 0.0012715
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015198, upper bound: 0.0015198
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015198, upper bound: 0.0015198
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0096273, 0.0098615
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052139
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013523, 0.0012468
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098286, 0.0096763
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012969, 0.0013457
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015005
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015005
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098541, 0.0096649
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012882, 0.0013605
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016286
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016286
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100656, 0.0100249
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0051735, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013538, 0.0014278
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015146, upper bound: 0.0015147
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015146, upper bound: 0.0015147
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100563, 0.0100516
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052004, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013824, 0.0014123
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015146, upper bound: 0.0015147
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015146, upper bound: 0.0015147
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100940, 0.0100193
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0051937, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013615, 0.0014256
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015677
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100846, 0.0100461
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052208, 0.0052236
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013903, 0.0014104
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015677
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0096399, 0.0098331
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052243, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013211, 0.0012662
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015198, upper bound: 0.0015198
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015198, upper bound: 0.0015198
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097671, 0.0096904
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052235, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012715, 0.0013071
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015698, upper bound: 0.0015677
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015677
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097001, 0.0097995
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052176, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013161, 0.0013226
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015004
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015020, upper bound: 0.0015004
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097256, 0.0097851
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052154, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013071, 0.0013374
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015004
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015004
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097096, 0.0097821
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0051950, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012919, 0.0013288
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097013, 0.0098087
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052220, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013187, 0.0013143
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097190, 0.0097722
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013025, 0.0013163
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
time: 1.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097445, 0.0097582
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052250, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012930, 0.0013310
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014928, upper bound: 0.0014928
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014928, upper bound: 0.0014928
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097106, 0.0097987
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013306, 0.0013026
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097360, 0.0097848
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013210, 0.0013173
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014928, upper bound: 0.0014928
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014941, upper bound: 0.0014928
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097933, 0.0096666
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052160, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0012547, 0.0013185
5: -0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670
6: -0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753
7: -0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175
8: -0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897
9: 1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015701, upper bound: 0.0015676
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015676
time: 1.36 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 4.08 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016254
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016254
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015198, upper bound: 0.0015198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015198, upper bound: 0.0015198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015005
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015005
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016286
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016286
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015146, upper bound: 0.0015147
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015146, upper bound: 0.0015147
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015146, upper bound: 0.0015147
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015146, upper bound: 0.0015147
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015677
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015677
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015198, upper bound: 0.0015198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015198, upper bound: 0.0015198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015698, upper bound: 0.0015677
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015677
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015004
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015020, upper bound: 0.0015004
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015004
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0015004
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015670
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0014928, upper bound: 0.0014928
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0014928, upper bound: 0.0014928
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0016252, upper bound: 0.0016256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0014928, upper bound: 0.0014928
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0014941, upper bound: 0.0014928
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015701, upper bound: 0.0015676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.08
Output dim: 9, lower bound: -0.0015670, upper bound: 0.0015676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015916
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015950
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015913, upper bound: 0.0015950
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016298, upper bound: 0.0016303
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016514
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016510
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016508, upper bound: 0.0016514
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016499, upper bound: 0.0016535
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015940, upper bound: 0.0015917
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015942, upper bound: 0.0015916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015991, upper bound: 0.0016004
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015991, upper bound: 0.0016004
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015951
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016642
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016642
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016643
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0016602, upper bound: 0.0016643
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.08
Output dim: 9, lower bound: -0.0015906, upper bound: 0.0015955

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.79 + 597.55 = 601.34 seconds
