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
execution time: IAR + RelationalAnalysis = 1.12 + 2.73 = 3.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0017606, upper bound: 0.0017606

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
time: 1.94 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.85
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.85
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0105900, 0.0105360
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015940, 0.0016209
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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017419, upper bound: 0.0017368
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017368, upper bound: 0.0017420
time: 1.41 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0105360, 0.0107039
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0016757, 0.0015940
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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017419, upper bound: 0.0017368
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017368, upper bound: 0.0017419
time: 1.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.10
Output dim: 9, lower bound: -0.0017419, upper bound: 0.0017368
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.10
Output dim: 9, lower bound: -0.0017368, upper bound: 0.0017420
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.10
Output dim: 9, lower bound: -0.0017419, upper bound: 0.0017368
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.10
Output dim: 9, lower bound: -0.0017368, upper bound: 0.0017419

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101863, 0.0102785
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015137, 0.0014938
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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017181, upper bound: 0.0017134
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017182, upper bound: 0.0017136
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0103140, 0.0101323
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014669, 0.0015358
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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017136, upper bound: 0.0017182
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017134, upper bound: 0.0017181
time: 2.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101323, 0.0104481
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015999, 0.0014669
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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017181, upper bound: 0.0017134
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017182, upper bound: 0.0017136
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0102785, 0.0103019
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015531, 0.0015137
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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017136, upper bound: 0.0017182
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017134, upper bound: 0.0017181
time: 1.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.06
Output dim: 9, lower bound: -0.0017181, upper bound: 0.0017134
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.06
Output dim: 9, lower bound: -0.0017182, upper bound: 0.0017136
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.06
Output dim: 9, lower bound: -0.0017136, upper bound: 0.0017182
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.06
Output dim: 9, lower bound: -0.0017134, upper bound: 0.0017181
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.06
Output dim: 9, lower bound: -0.0017181, upper bound: 0.0017134
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.06
Output dim: 9, lower bound: -0.0017182, upper bound: 0.0017136
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.06
Output dim: 9, lower bound: -0.0017136, upper bound: 0.0017182
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.06
Output dim: 9, lower bound: -0.0017134, upper bound: 0.0017181

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100044, 0.0101361
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014586, 0.0014140
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017088
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017089
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100318, 0.0100966
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014339, 0.0014309
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017089
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017090
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101321, 0.0099885
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014108, 0.0014560
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017090, upper bound: 0.0017134
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017090, upper bound: 0.0017134
time: 2.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101646, 0.0099504
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013871, 0.0014728
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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017133
time: 2.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017133
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099504, 0.0103091
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015479, 0.0013871
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017133, upper bound: 0.0017088
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017133, upper bound: 0.0017089
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099885, 0.0102696
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015232, 0.0014108
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017134, upper bound: 0.0017090
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017090
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100966, 0.0101614
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015001, 0.0014339
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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017090, upper bound: 0.0017134
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017134
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101361, 0.0101234
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014764, 0.0014586
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017133
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017133
time: 2.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017088
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017089
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017089
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017090
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017090, upper bound: 0.0017134
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017090, upper bound: 0.0017134
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017133
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017133
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017133, upper bound: 0.0017088
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017133, upper bound: 0.0017089
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017134, upper bound: 0.0017090
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017090
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017090, upper bound: 0.0017134
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017134
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017133
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 9, lower bound: -0.0017088, upper bound: 0.0017133

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099954, 0.0101291
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014498, 0.0014020
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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016890
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016890
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099977, 0.0101272
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014466, 0.0014056
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016891
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016891
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100228, 0.0100897
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014244, 0.0014189
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016893
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100252, 0.0100876
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014219, 0.0014230
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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
time: 2.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101231, 0.0099820
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014024, 0.0014439
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101255, 0.0099795
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013987, 0.0014482
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101556, 0.0099433
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013783, 0.0014607
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101582, 0.0099414
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013751, 0.0014648
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099414, 0.0103014
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015381, 0.0013751
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016890
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016890
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099434, 0.0102995
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015349, 0.0013783
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016891
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016891
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099795, 0.0102620
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015127, 0.0013987
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099820, 0.0102599
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015102, 0.0014024
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016893
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016893
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100876, 0.0101543
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014907, 0.0014219
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016893, upper bound: 0.0016935
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100897, 0.0101518
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014871, 0.0014244
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101271, 0.0101157
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014667, 0.0014466
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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101291, 0.0101138
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014635, 0.0014498
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
time: 1.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016890
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016891
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016891
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016893
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016890
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016890
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016891
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016891
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016893
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016893
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016935, upper bound: 0.0016893
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016893, upper bound: 0.0016935
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.74
Output dim: 9, lower bound: -0.0016891, upper bound: 0.0016935

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099655, 0.0101007
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014428, 0.0013969
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016778
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099954, 0.0100992
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014498, 0.0013950
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099678, 0.0100985
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014395, 0.0014005
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099977, 0.0100973
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014466, 0.0013986
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016778
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099930, 0.0100619
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014174, 0.0014135
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016874, upper bound: 0.0016784
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100228, 0.0100598
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014244, 0.0014119
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016874, upper bound: 0.0016784
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099954, 0.0100599
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014149, 0.0014175
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
time: 2.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100252, 0.0100577
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014219, 0.0014160
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100932, 0.0099527
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013954, 0.0014385
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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101231, 0.0099521
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014024, 0.0014369
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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100956, 0.0099503
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013917, 0.0014426
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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101255, 0.0099496
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013987, 0.0014412
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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101258, 0.0099141
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013713, 0.0014549
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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101556, 0.0099135
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013783, 0.0014537
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101284, 0.0099120
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013681, 0.0014591
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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 2.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101582, 0.0099115
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013751, 0.0014577
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099116, 0.0102730
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015311, 0.0013687
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099414, 0.0102715
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015381, 0.0013681
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099135, 0.0102709
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015279, 0.0013716
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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016778
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099434, 0.0102696
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015349, 0.0013713
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099496, 0.0102342
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015057, 0.0013925
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016874, upper bound: 0.0016784
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099795, 0.0102321
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015127, 0.0013917
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099521, 0.0102322
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015032, 0.0013957
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099820, 0.0102301
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0015102, 0.0013954
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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100577, 0.0101250
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014837, 0.0014157
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100876, 0.0101245
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014907, 0.0014149
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100598, 0.0101226
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014801, 0.0014179
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100897, 0.0101219
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014871, 0.0014174
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100973, 0.0100864
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014597, 0.0014396
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101271, 0.0100858
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014667, 0.0014395
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0100992, 0.0100843
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014565, 0.0014428
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 2.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0101291, 0.0100839
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014635, 0.0014428
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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
time: 2.23 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016778
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016778
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016874, upper bound: 0.0016784
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016874, upper bound: 0.0016784
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016778
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016777
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016830
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016874, upper bound: 0.0016784
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016784
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016833
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016821
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016822
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 9, lower bound: -0.0016778, upper bound: 0.0016874

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097663, 0.0099418
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014034, 0.0013571
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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098050, 0.0099015
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014030, 0.0013570
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015897
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015897
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097946, 0.0099348
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014119, 0.0013553
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098333, 0.0099000
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014115, 0.0013559
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015897
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015896
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097686, 0.0099399
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013995, 0.0013607
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098073, 0.0098993
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013998, 0.0013610
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015897
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015897
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097969, 0.0099330
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014080, 0.0013589
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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098356, 0.0098981
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0014082, 0.0013601
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015897
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015897
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097938, 0.0099125
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013810, 0.0013738
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098224, 0.0098627
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013776, 0.0013717
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015895
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015895
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098220, 0.0099055
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013895, 0.0013721
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098507, 0.0098606
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013861, 0.0013700
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015894
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015894
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0097962, 0.0099106
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013778, 0.0013777
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015857
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098249, 0.0098606
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013751, 0.0013763
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015895
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015895
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098244, 0.0099037
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013862, 0.0013762
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015857
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015858
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098532, 0.0098585
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013836, 0.0013751
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015894
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015895
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098940, 0.0097978
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013541, 0.0013987
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099330, 0.0097535
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013556, 0.0013998
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015907
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015907
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099223, 0.0097923
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013625, 0.0013972
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099612, 0.0097529
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013640, 0.0013987
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015907
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015907
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0098964, 0.0097958
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013499, 0.0014029
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099353, 0.0097511
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013520, 0.0014038
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015907
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015907
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099247, 0.0097903
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013583, 0.0014014
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099635, 0.0097504
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013604, 0.0014028
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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015907
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015907
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099266, 0.0097718
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013332, 0.0014152
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099559, 0.0097148
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013316, 0.0014141
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015904
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015904
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099548, 0.0097665
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013416, 0.0014140
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099842, 0.0097143
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013400, 0.0014130
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015904
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015904
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099292, 0.0097699
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013296, 0.0014193
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015868
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099583, 0.0097127
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013284, 0.0014181
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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015904
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015858, upper bound: 0.0015904
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741
1: 0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977
2: 0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0099574, 0.0097644
3: -0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259
4: 0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0013380, 0.0014180
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

Time for backsubstitution: 1.12 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.86 + 597.17 = 601.03 seconds
