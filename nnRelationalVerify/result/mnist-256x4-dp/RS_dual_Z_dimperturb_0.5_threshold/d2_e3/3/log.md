## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00031548


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0019769, 0.0019769)
1: (-0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005574, 0.0005574)
2: (-0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0041124, 0.0041124)
3: (0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0005442, 0.0005442)
4: (0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0030734, 0.0030734)
5: (0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0008539, 0.0008539)
6: (0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0007751, 0.0007751)
7: (-0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0028924, 0.0028924)
8: (-0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0022512, 0.0022512)
9: (-0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001942, 0.0001942)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 1.76 = 3.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0005530, upper bound: 0.0005531

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005271, upper bound: 0.0005256
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005255, upper bound: 0.0005271
time: 0.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.79 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 5, lower bound: -0.0005271, upper bound: 0.0005256
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 5, lower bound: -0.0005255, upper bound: 0.0005271

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0019719, 0.0019762
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005560, 0.0005572
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0041019, 0.0041109
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0005428, 0.0005440
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0030722, 0.0030655
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0008536, 0.0008517
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0007748, 0.0007731
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0028913, 0.0028850
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0022454, 0.0022503
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001941, 0.0001937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004985, upper bound: 0.0005078
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005090, upper bound: 0.0004952
time: 0.77 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0019769, 0.0019719
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005574, 0.0005560
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0041124, 0.0041019
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0005442, 0.0005428
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0030655, 0.0030734
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0008517, 0.0008539
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0007731, 0.0007751
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0028850, 0.0028924
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0022512, 0.0022454
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001937, 0.0001942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004951, upper bound: 0.0005091
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005078, upper bound: 0.0004985
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 5, lower bound: -0.0004985, upper bound: 0.0005078
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 5, lower bound: -0.0005090, upper bound: 0.0004952
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 5, lower bound: -0.0004951, upper bound: 0.0005091
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 5, lower bound: -0.0005078, upper bound: 0.0004985

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017883, 0.0017770
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005042, 0.0005010
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0037201, 0.0036966
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004923, 0.0004892
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027626, 0.0027802
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007675, 0.0007724
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006967, 0.0007011
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0025999, 0.0026165
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020364, 0.0020235
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001746, 0.0001757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004795, upper bound: 0.0004906
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004795, upper bound: 0.0004907
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017727, 0.0017919
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004998, 0.0005052
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0036876, 0.0037275
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004880, 0.0004933
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027857, 0.0027559
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007739, 0.0007657
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0007025, 0.0006950
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0026217, 0.0025936
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020186, 0.0020404
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001760, 0.0001742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004908, upper bound: 0.0004785
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004907, upper bound: 0.0004786
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017941, 0.0017727
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005058, 0.0004998
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0037321, 0.0036876
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004939, 0.0004880
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027559, 0.0027891
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007657, 0.0007749
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006950, 0.0007034
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0025936, 0.0026249
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020429, 0.0020186
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001742, 0.0001763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004786, upper bound: 0.0004908
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004786, upper bound: 0.0004908
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017785, 0.0017883
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005014, 0.0005042
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0036996, 0.0037201
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004896, 0.0004923
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027802, 0.0027648
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007724, 0.0007682
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0007011, 0.0006973
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0026165, 0.0026020
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020252, 0.0020364
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001757, 0.0001747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004906, upper bound: 0.0004796
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004906, upper bound: 0.0004795
time: 0.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 5, lower bound: -0.0004795, upper bound: 0.0004906
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 5, lower bound: -0.0004795, upper bound: 0.0004907
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 5, lower bound: -0.0004908, upper bound: 0.0004785
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 5, lower bound: -0.0004907, upper bound: 0.0004786
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 5, lower bound: -0.0004786, upper bound: 0.0004908
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 5, lower bound: -0.0004786, upper bound: 0.0004908
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 5, lower bound: -0.0004906, upper bound: 0.0004796
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 5, lower bound: -0.0004906, upper bound: 0.0004795

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017768, 0.0017688
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005010, 0.0004987
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0036962, 0.0036794
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004891, 0.0004869
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027497, 0.0027623
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007640, 0.0007674
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006934, 0.0006966
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0025878, 0.0025996
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020233, 0.0020141
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001738, 0.0001746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004615, upper bound: 0.0004741
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004630, upper bound: 0.0004710
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017779, 0.0017655
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005013, 0.0004978
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0036984, 0.0036727
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004894, 0.0004860
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027447, 0.0027639
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007626, 0.0007679
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006922, 0.0006970
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0025831, 0.0026012
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020245, 0.0020104
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001735, 0.0001747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004614, upper bound: 0.0004741
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004630, upper bound: 0.0004711
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017612, 0.0017817
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004966, 0.0005023
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0036637, 0.0037063
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004848, 0.0004905
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027698, 0.0027380
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007695, 0.0007607
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006985, 0.0006905
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0026067, 0.0025768
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020055, 0.0020288
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001750, 0.0001730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004713, upper bound: 0.0004616
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004743, upper bound: 0.0004597
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017646, 0.0017804
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004975, 0.0005020
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0036707, 0.0037036
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004858, 0.0004901
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027678, 0.0027432
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007690, 0.0007621
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006980, 0.0006918
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0026048, 0.0025817
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020093, 0.0020273
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001749, 0.0001734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004713, upper bound: 0.0004616
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004743, upper bound: 0.0004602
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017823, 0.0017646
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005025, 0.0004975
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0037076, 0.0036707
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004906, 0.0004858
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027432, 0.0027708
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007621, 0.0007698
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006918, 0.0006988
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0025817, 0.0026077
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020296, 0.0020093
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001734, 0.0001751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004601, upper bound: 0.0004743
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004616, upper bound: 0.0004712
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017834, 0.0017612
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0005028, 0.0004966
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0037098, 0.0036637
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004909, 0.0004848
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027380, 0.0027725
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007607, 0.0007703
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006905, 0.0006992
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0025768, 0.0026092
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020307, 0.0020055
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001730, 0.0001752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004597, upper bound: 0.0004743
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004616, upper bound: 0.0004713
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017667, 0.0017779
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004981, 0.0005013
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0036751, 0.0036984
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004863, 0.0004894
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027639, 0.0027466
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007679, 0.0007631
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006970, 0.0006926
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0026012, 0.0025848
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020118, 0.0020245
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001747, 0.0001736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004711, upper bound: 0.0004631
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004741, upper bound: 0.0004614
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0017701, 0.0017768
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004990, 0.0005010
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0036821, 0.0036962
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004873, 0.0004891
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0027623, 0.0027518
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0007674, 0.0007645
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006966, 0.0006940
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0025996, 0.0025897
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0020156, 0.0020233
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001746, 0.0001739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004709, upper bound: 0.0004631
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004741, upper bound: 0.0004615
time: 0.89 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004615, upper bound: 0.0004741
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004630, upper bound: 0.0004710
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004614, upper bound: 0.0004741
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004630, upper bound: 0.0004711
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004713, upper bound: 0.0004616
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004743, upper bound: 0.0004597
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004713, upper bound: 0.0004616
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004743, upper bound: 0.0004602
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004601, upper bound: 0.0004743
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004616, upper bound: 0.0004712
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004597, upper bound: 0.0004743
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004616, upper bound: 0.0004713
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004711, upper bound: 0.0004631
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004741, upper bound: 0.0004614
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004709, upper bound: 0.0004631
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 5, lower bound: -0.0004741, upper bound: 0.0004615

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015940, 0.0015786
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004494, 0.0004451
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033158, 0.0032838
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004388, 0.0004346
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024541, 0.0024780
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006818, 0.0006885
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006189, 0.0006249
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023096, 0.0023321
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018150, 0.0017975
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001551, 0.0001566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004643
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004498, upper bound: 0.0004687
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015867, 0.0015788
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004473, 0.0004451
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033006, 0.0032843
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004368, 0.0004346
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024545, 0.0024667
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006819, 0.0006853
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006190, 0.0006221
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023099, 0.0023214
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018068, 0.0017978
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001551, 0.0001559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004577, upper bound: 0.0004617
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004510, upper bound: 0.0004653
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015937, 0.0015754
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004493, 0.0004442
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033152, 0.0032771
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004387, 0.0004337
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024491, 0.0024775
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006804, 0.0006883
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006176, 0.0006248
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023049, 0.0023316
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018147, 0.0017939
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001548, 0.0001566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004561, upper bound: 0.0004643
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004492, upper bound: 0.0004686
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015877, 0.0015769
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004476, 0.0004446
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033028, 0.0032803
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004371, 0.0004341
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024515, 0.0024683
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006811, 0.0006858
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006182, 0.0006225
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023071, 0.0023229
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018079, 0.0017956
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001549, 0.0001560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004577, upper bound: 0.0004618
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004510, upper bound: 0.0004655
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015740, 0.0015915
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004438, 0.0004487
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032742, 0.0033107
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004333, 0.0004381
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024742, 0.0024469
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006874, 0.0006798
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006240, 0.0006171
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023285, 0.0023028
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017923, 0.0018123
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001564, 0.0001546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004657, upper bound: 0.0004502
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004622, upper bound: 0.0004562
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015711, 0.0015965
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004429, 0.0004501
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032681, 0.0033210
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004325, 0.0004395
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024819, 0.0024424
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006895, 0.0006786
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006259, 0.0006159
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023357, 0.0022986
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017890, 0.0018179
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001568, 0.0001543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004688, upper bound: 0.0004480
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004647, upper bound: 0.0004543
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015755, 0.0015902
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004442, 0.0004483
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032773, 0.0033080
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004337, 0.0004378
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024722, 0.0024493
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006868, 0.0006805
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006234, 0.0006177
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023266, 0.0023050
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017940, 0.0018108
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001562, 0.0001548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004656, upper bound: 0.0004503
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004621, upper bound: 0.0004562
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015744, 0.0015965
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004439, 0.0004501
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032751, 0.0033210
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004334, 0.0004395
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024819, 0.0024476
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006896, 0.0006800
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006259, 0.0006172
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023358, 0.0023034
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017928, 0.0018179
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001568, 0.0001547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004688, upper bound: 0.0004488
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004647, upper bound: 0.0004547
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015999, 0.0015744
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004511, 0.0004439
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033280, 0.0032751
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004404, 0.0004334
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024476, 0.0024872
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006800, 0.0006910
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006172, 0.0006272
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023034, 0.0023407
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018218, 0.0017928
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001547, 0.0001572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004547, upper bound: 0.0004647
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004488, upper bound: 0.0004688
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015926, 0.0015755
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004490, 0.0004442
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033129, 0.0032773
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004384, 0.0004337
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024493, 0.0024758
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006805, 0.0006879
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006177, 0.0006244
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023050, 0.0023300
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018135, 0.0017940
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001548, 0.0001565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004561, upper bound: 0.0004621
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004502, upper bound: 0.0004657
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015996, 0.0015711
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004510, 0.0004429
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033274, 0.0032681
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004403, 0.0004325
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024424, 0.0024867
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006786, 0.0006909
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006159, 0.0006271
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022986, 0.0023403
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018214, 0.0017890
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001543, 0.0001571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004543, upper bound: 0.0004646
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004480, upper bound: 0.0004688
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015936, 0.0015740
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004493, 0.0004438
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033150, 0.0032742
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004387, 0.0004333
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024469, 0.0024775
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006798, 0.0006883
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006171, 0.0006248
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023028, 0.0023316
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018147, 0.0017923
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001546, 0.0001566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004561, upper bound: 0.0004622
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004501, upper bound: 0.0004657
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015799, 0.0015877
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004454, 0.0004476
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032865, 0.0033028
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004349, 0.0004371
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024683, 0.0024561
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006858, 0.0006824
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006225, 0.0006194
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023229, 0.0023115
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017990, 0.0018079
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001560, 0.0001552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004655, upper bound: 0.0004510
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004618, upper bound: 0.0004577
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015770, 0.0015937
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004446, 0.0004493
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032804, 0.0033152
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004341, 0.0004387
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024775, 0.0024516
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006883, 0.0006811
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006248, 0.0006183
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023316, 0.0023072
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017957, 0.0018147
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001566, 0.0001549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004686, upper bound: 0.0004492
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004643, upper bound: 0.0004561
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015814, 0.0015867
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004459, 0.0004473
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032896, 0.0033006
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004353, 0.0004368
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024667, 0.0024584
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006853, 0.0006830
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006221, 0.0006200
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023214, 0.0023137
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018007, 0.0018068
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001559, 0.0001554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004653, upper bound: 0.0004510
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004617, upper bound: 0.0004577
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015803, 0.0015940
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004455, 0.0004494
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032874, 0.0033158
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004350, 0.0004388
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024780, 0.0024568
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006885, 0.0006826
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006249, 0.0006196
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023321, 0.0023121
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017995, 0.0018150
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001566, 0.0001553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004686, upper bound: 0.0004498
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004643, upper bound: 0.0004562
time: 0.86 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004643
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004498, upper bound: 0.0004687
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004577, upper bound: 0.0004617
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004510, upper bound: 0.0004653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004561, upper bound: 0.0004643
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004492, upper bound: 0.0004686
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004577, upper bound: 0.0004618
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004510, upper bound: 0.0004655
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004657, upper bound: 0.0004502
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004622, upper bound: 0.0004562
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004688, upper bound: 0.0004480
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004647, upper bound: 0.0004543
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004656, upper bound: 0.0004503
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004621, upper bound: 0.0004562
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004688, upper bound: 0.0004488
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004647, upper bound: 0.0004547
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004547, upper bound: 0.0004647
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004488, upper bound: 0.0004688
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004561, upper bound: 0.0004621
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004502, upper bound: 0.0004657
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004543, upper bound: 0.0004646
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004480, upper bound: 0.0004688
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004561, upper bound: 0.0004622
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004501, upper bound: 0.0004657
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004655, upper bound: 0.0004510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004618, upper bound: 0.0004577
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004686, upper bound: 0.0004492
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004643, upper bound: 0.0004561
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004653, upper bound: 0.0004510
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004617, upper bound: 0.0004577
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004686, upper bound: 0.0004498
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 5, lower bound: -0.0004643, upper bound: 0.0004562

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015828, 0.0015708
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004462, 0.0004429
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032925, 0.0032676
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004357, 0.0004324
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024420, 0.0024606
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006785, 0.0006836
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006158, 0.0006205
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022982, 0.0023157
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018023, 0.0017887
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001543, 0.0001555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015854, 0.0015674
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004470, 0.0004419
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032979, 0.0032605
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004364, 0.0004315
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024367, 0.0024647
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006770, 0.0006848
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006145, 0.0006216
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022932, 0.0023195
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018053, 0.0017848
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001540, 0.0001558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015755, 0.0015706
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004442, 0.0004428
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032773, 0.0032671
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004337, 0.0004324
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024417, 0.0024493
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006784, 0.0006805
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006157, 0.0006177
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022979, 0.0023050
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017940, 0.0017884
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001543, 0.0001548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015781, 0.0015676
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004449, 0.0004420
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032828, 0.0032610
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004344, 0.0004315
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024371, 0.0024534
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006771, 0.0006816
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006146, 0.0006187
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022936, 0.0023089
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017970, 0.0017851
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001540, 0.0001550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015825, 0.0015674
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004462, 0.0004419
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032919, 0.0032605
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004356, 0.0004315
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024367, 0.0024601
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006770, 0.0006835
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006145, 0.0006204
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022932, 0.0023153
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018020, 0.0017848
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001540, 0.0001555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015853, 0.0015642
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004470, 0.0004410
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032978, 0.0032538
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004364, 0.0004306
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024317, 0.0024646
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006756, 0.0006847
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006132, 0.0006215
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022885, 0.0023194
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018052, 0.0017811
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001537, 0.0001557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015765, 0.0015688
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004445, 0.0004423
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032795, 0.0032635
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004340, 0.0004319
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024390, 0.0024509
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006776, 0.0006809
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006151, 0.0006181
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022953, 0.0023066
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017952, 0.0017865
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001541, 0.0001549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015792, 0.0015657
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004452, 0.0004414
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032851, 0.0032570
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004347, 0.0004310
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024341, 0.0024550
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006763, 0.0006821
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006138, 0.0006191
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022907, 0.0023105
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017982, 0.0017829
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001538, 0.0001551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015628, 0.0015828
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004406, 0.0004463
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032509, 0.0032926
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004302, 0.0004357
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024607, 0.0024295
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006836, 0.0006750
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006205, 0.0006127
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023158, 0.0022865
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017796, 0.0018024
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001555, 0.0001535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015659, 0.0015803
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004415, 0.0004456
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032574, 0.0032874
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004311, 0.0004350
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024568, 0.0024343
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006826, 0.0006763
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006196, 0.0006139
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023121, 0.0022910
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017831, 0.0017995
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001553, 0.0001538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015599, 0.0015880
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004398, 0.0004477
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032449, 0.0033035
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004294, 0.0004372
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024688, 0.0024250
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006859, 0.0006737
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006226, 0.0006115
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023234, 0.0022822
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017762, 0.0018083
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001560, 0.0001532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015631, 0.0015853
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004407, 0.0004470
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032517, 0.0032977
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004303, 0.0004364
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024645, 0.0024301
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006847, 0.0006752
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006215, 0.0006128
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023194, 0.0022870
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017800, 0.0018052
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001557, 0.0001536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015643, 0.0015815
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004410, 0.0004459
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032540, 0.0032899
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004306, 0.0004354
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024587, 0.0024319
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006831, 0.0006756
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006200, 0.0006133
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023139, 0.0022887
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017813, 0.0018009
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001554, 0.0001537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015670, 0.0015790
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004418, 0.0004452
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032598, 0.0032847
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004314, 0.0004347
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024548, 0.0024361
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006820, 0.0006768
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006191, 0.0006144
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023102, 0.0022927
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017844, 0.0017980
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001551, 0.0001539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015632, 0.0015881
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004407, 0.0004477
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032518, 0.0033035
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004303, 0.0004372
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024688, 0.0024302
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006859, 0.0006752
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006226, 0.0006129
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023234, 0.0022871
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017800, 0.0018083
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001560, 0.0001536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015665, 0.0015853
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004417, 0.0004470
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032587, 0.0032978
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004312, 0.0004364
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024645, 0.0024354
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006847, 0.0006766
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006215, 0.0006142
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023194, 0.0022920
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017838, 0.0018052
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001557, 0.0001539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015887, 0.0015665
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004479, 0.0004417
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033048, 0.0032587
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004373, 0.0004312
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024354, 0.0024698
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006766, 0.0006862
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006142, 0.0006228
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022920, 0.0023243
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018090, 0.0017838
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001539, 0.0001561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015913, 0.0015632
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004486, 0.0004407
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033102, 0.0032518
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004381, 0.0004303
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024302, 0.0024739
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006752, 0.0006873
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006129, 0.0006239
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022871, 0.0023282
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018120, 0.0017800
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001536, 0.0001563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015814, 0.0015670
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004459, 0.0004418
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032896, 0.0032598
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004353, 0.0004314
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024361, 0.0024584
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006768, 0.0006830
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006144, 0.0006200
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022927, 0.0023137
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018007, 0.0017844
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001539, 0.0001554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015840, 0.0015643
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004466, 0.0004410
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032951, 0.0032540
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004361, 0.0004306
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024319, 0.0024626
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006756, 0.0006842
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006133, 0.0006210
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022887, 0.0023175
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018037, 0.0017813
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001537, 0.0001556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015884, 0.0015631
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004478, 0.0004407
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033042, 0.0032517
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004373, 0.0004303
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024301, 0.0024693
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006752, 0.0006861
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006128, 0.0006227
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022870, 0.0023239
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018087, 0.0017800
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001536, 0.0001560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015912, 0.0015599
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004486, 0.0004398
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033101, 0.0032449
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004380, 0.0004294
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024250, 0.0024737
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006737, 0.0006873
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006115, 0.0006238
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022822, 0.0023281
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018119, 0.0017762
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001532, 0.0001563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015824, 0.0015659
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004461, 0.0004415
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032918, 0.0032574
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004356, 0.0004311
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024343, 0.0024601
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006763, 0.0006835
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006139, 0.0006204
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022910, 0.0023152
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018019, 0.0017831
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001538, 0.0001555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015851, 0.0015628
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004469, 0.0004406
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032973, 0.0032509
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004364, 0.0004302
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024295, 0.0024642
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006750, 0.0006846
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006127, 0.0006214
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022865, 0.0023191
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018050, 0.0017796
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001535, 0.0001557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015687, 0.0015792
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004423, 0.0004452
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032632, 0.0032851
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004318, 0.0004347
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024550, 0.0024387
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006821, 0.0006775
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006191, 0.0006150
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023105, 0.0022951
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017863, 0.0017982
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001551, 0.0001541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015718, 0.0015765
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004431, 0.0004445
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032696, 0.0032795
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004327, 0.0004340
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024509, 0.0024435
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006809, 0.0006789
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006181, 0.0006162
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023066, 0.0022996
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017898, 0.0017952
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001549, 0.0001544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015658, 0.0015853
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004415, 0.0004470
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032571, 0.0032978
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004310, 0.0004364
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024646, 0.0024342
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006847, 0.0006763
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006215, 0.0006139
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023194, 0.0022908
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017830, 0.0018052
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001557, 0.0001538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015691, 0.0015825
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004424, 0.0004462
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032640, 0.0032919
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004319, 0.0004356
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024601, 0.0024393
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006835, 0.0006777
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006204, 0.0006152
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023153, 0.0022956
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017867, 0.0018020
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001555, 0.0001541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015702, 0.0015781
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004427, 0.0004449
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032663, 0.0032828
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004322, 0.0004344
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024534, 0.0024411
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006816, 0.0006782
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006187, 0.0006156
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023089, 0.0022973
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017880, 0.0017970
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001550, 0.0001543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015729, 0.0015755
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004435, 0.0004442
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032720, 0.0032773
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004330, 0.0004337
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024493, 0.0024453
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006805, 0.0006794
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006177, 0.0006167
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023050, 0.0023013
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017911, 0.0017940
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001548, 0.0001545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015691, 0.0015854
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004424, 0.0004470
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032641, 0.0032979
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004319, 0.0004364
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024647, 0.0024394
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006848, 0.0006777
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006216, 0.0006152
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023195, 0.0022957
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017868, 0.0018053
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001558, 0.0001542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015724, 0.0015828
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004433, 0.0004462
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032710, 0.0032925
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004329, 0.0004357
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024606, 0.0024446
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006836, 0.0006792
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006205, 0.0006165
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023157, 0.0023006
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017906, 0.0018023
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001555, 0.0001545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
time: 0.83 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003349, upper bound: 0.0003409
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003331, upper bound: 0.0003419
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003358, upper bound: 0.0003391
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003343, upper bound: 0.0003401
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003343
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003359
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003419, upper bound: 0.0003331
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003349

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015601, 0.0015551
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004399, 0.0004384
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032454, 0.0032350
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004295, 0.0004281
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024176, 0.0024254
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006717, 0.0006738
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006097, 0.0006116
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022753, 0.0022826
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017765, 0.0017708
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001528, 0.0001533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015828, 0.0015481
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004462, 0.0004365
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032925, 0.0032205
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004357, 0.0004262
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024068, 0.0024606
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006687, 0.0006836
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006070, 0.0006205
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022650, 0.0023157
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018023, 0.0017629
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001521, 0.0001555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015627, 0.0015517
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004406, 0.0004375
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032508, 0.0032279
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004302, 0.0004272
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024123, 0.0024295
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006702, 0.0006750
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006084, 0.0006127
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022703, 0.0022864
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017795, 0.0017670
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001524, 0.0001535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015854, 0.0015447
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004470, 0.0004355
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032979, 0.0032134
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004364, 0.0004252
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024015, 0.0024647
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006672, 0.0006848
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006056, 0.0006216
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022601, 0.0023195
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018053, 0.0017590
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001518, 0.0001558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015528, 0.0015547
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004378, 0.0004383
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032302, 0.0032340
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004275, 0.0004280
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024169, 0.0024141
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006715, 0.0006707
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006095, 0.0006088
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022746, 0.0022719
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017682, 0.0017703
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001527, 0.0001526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015755, 0.0015479
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004442, 0.0004364
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032773, 0.0032200
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004337, 0.0004261
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024064, 0.0024493
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006686, 0.0006805
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006069, 0.0006177
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022647, 0.0023050
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017940, 0.0017626
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001521, 0.0001548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015555, 0.0015517
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004385, 0.0004375
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032357, 0.0032279
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004282, 0.0004272
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024123, 0.0024182
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006702, 0.0006718
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006084, 0.0006098
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022703, 0.0022758
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017712, 0.0017670
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001524, 0.0001528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015781, 0.0015450
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004449, 0.0004356
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032828, 0.0032139
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004344, 0.0004253
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024019, 0.0024534
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006673, 0.0006816
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006057, 0.0006187
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022604, 0.0023089
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017970, 0.0017593
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001518, 0.0001550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015598, 0.0015511
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004398, 0.0004373
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032448, 0.0032267
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004294, 0.0004270
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024114, 0.0024249
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006700, 0.0006737
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006081, 0.0006115
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022694, 0.0022821
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017762, 0.0017663
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001524, 0.0001532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015825, 0.0015448
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004462, 0.0004355
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032919, 0.0032134
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004356, 0.0004252
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024015, 0.0024601
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006672, 0.0006835
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006056, 0.0006204
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022601, 0.0023153
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018020, 0.0017590
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001518, 0.0001555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015627, 0.0015479
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004406, 0.0004364
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032507, 0.0032200
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004302, 0.0004261
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024065, 0.0024293
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006686, 0.0006749
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006069, 0.0006126
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022647, 0.0022863
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017794, 0.0017627
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001521, 0.0001535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015853, 0.0015415
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004470, 0.0004346
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032978, 0.0032067
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004364, 0.0004244
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023965, 0.0024646
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006658, 0.0006847
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006044, 0.0006215
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022554, 0.0023194
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018052, 0.0017554
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001514, 0.0001557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015539, 0.0015533
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004381, 0.0004379
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032324, 0.0032311
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004278, 0.0004276
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024147, 0.0024157
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006709, 0.0006711
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006090, 0.0006092
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022725, 0.0022734
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017694, 0.0017687
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001526, 0.0001527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015765, 0.0015462
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004445, 0.0004359
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032795, 0.0032164
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004340, 0.0004256
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024037, 0.0024509
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006678, 0.0006809
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006062, 0.0006181
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022622, 0.0023066
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017952, 0.0017607
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001519, 0.0001549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015566, 0.0015499
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004388, 0.0004370
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032379, 0.0032241
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004285, 0.0004267
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024095, 0.0024198
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006694, 0.0006723
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006076, 0.0006102
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022676, 0.0022773
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017725, 0.0017649
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001523, 0.0001529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015792, 0.0015431
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004452, 0.0004350
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032851, 0.0032099
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004347, 0.0004248
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023989, 0.0024550
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006665, 0.0006821
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006050, 0.0006191
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022576, 0.0023105
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017982, 0.0017571
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001516, 0.0001551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015401, 0.0015684
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004342, 0.0004422
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032038, 0.0032625
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004240, 0.0004317
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024382, 0.0023943
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006774, 0.0006652
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006149, 0.0006038
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022946, 0.0022533
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017538, 0.0017859
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001541, 0.0001513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015628, 0.0015602
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004406, 0.0004399
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032509, 0.0032455
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004302, 0.0004295
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024255, 0.0024295
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006739, 0.0006750
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006117, 0.0006127
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022826, 0.0022865
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017796, 0.0017766
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001533, 0.0001535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015432, 0.0015658
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004351, 0.0004414
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032102, 0.0032571
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004248, 0.0004310
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024341, 0.0023991
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006763, 0.0006666
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006139, 0.0006050
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022908, 0.0022579
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017573, 0.0017829
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001538, 0.0001516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015659, 0.0015577
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004415, 0.0004392
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032574, 0.0032403
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004311, 0.0004288
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024216, 0.0024343
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006728, 0.0006763
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006107, 0.0006139
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022790, 0.0022910
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017831, 0.0017737
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001530, 0.0001538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015372, 0.0015717
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004334, 0.0004431
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031977, 0.0032694
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004232, 0.0004327
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024433, 0.0023898
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006788, 0.0006640
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006162, 0.0006027
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022995, 0.0022491
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017504, 0.0017897
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001544, 0.0001510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015599, 0.0015654
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004398, 0.0004413
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032449, 0.0032563
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004294, 0.0004309
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024336, 0.0024250
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006761, 0.0006737
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006137, 0.0006115
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022903, 0.0022822
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017762, 0.0017825
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001538, 0.0001532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015405, 0.0015685
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004343, 0.0004422
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032046, 0.0032628
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004241, 0.0004318
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024384, 0.0023949
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006775, 0.0006654
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006149, 0.0006040
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022948, 0.0022539
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017542, 0.0017861
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001541, 0.0001513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015631, 0.0015626
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004407, 0.0004406
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032517, 0.0032506
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004303, 0.0004302
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024293, 0.0024301
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006749, 0.0006752
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006126, 0.0006128
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022862, 0.0022870
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017800, 0.0017794
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001535, 0.0001536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015416, 0.0015664
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004346, 0.0004416
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032069, 0.0032584
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004244, 0.0004312
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024351, 0.0023967
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006765, 0.0006659
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006141, 0.0006044
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022917, 0.0022555
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017555, 0.0017836
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001539, 0.0001515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015643, 0.0015589
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004410, 0.0004395
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032540, 0.0032428
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004306, 0.0004291
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024235, 0.0024319
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006733, 0.0006756
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006112, 0.0006133
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022808, 0.0022887
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017813, 0.0017751
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001531, 0.0001537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015444, 0.0015637
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004354, 0.0004409
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032126, 0.0032528
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004251, 0.0004305
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024309, 0.0024009
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006754, 0.0006670
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006130, 0.0006055
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022878, 0.0022595
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017586, 0.0017806
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001536, 0.0001517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015670, 0.0015564
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004418, 0.0004388
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032598, 0.0032376
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004314, 0.0004284
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024196, 0.0024361
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006722, 0.0006768
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006102, 0.0006144
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022771, 0.0022927
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017844, 0.0017723
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001529, 0.0001539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015406, 0.0015721
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004343, 0.0004432
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032047, 0.0032702
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004241, 0.0004328
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024439, 0.0023950
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006790, 0.0006654
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006163, 0.0006040
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0023000, 0.0022539
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017542, 0.0017901
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001544, 0.0001513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015632, 0.0015654
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004407, 0.0004413
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032518, 0.0032564
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004303, 0.0004309
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024336, 0.0024302
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006761, 0.0006752
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006137, 0.0006129
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022903, 0.0022871
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017800, 0.0017825
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001538, 0.0001536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015439, 0.0015689
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004353, 0.0004423
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032116, 0.0032637
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004250, 0.0004319
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024391, 0.0024002
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006777, 0.0006668
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006151, 0.0006053
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022955, 0.0022588
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017580, 0.0017866
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001541, 0.0001517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015665, 0.0015627
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004417, 0.0004406
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032587, 0.0032506
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004312, 0.0004302
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024293, 0.0024354
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006749, 0.0006766
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006126, 0.0006142
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022863, 0.0022920
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017838, 0.0017794
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001535, 0.0001539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015661, 0.0015509
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004415, 0.0004373
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032578, 0.0032262
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004311, 0.0004269
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024110, 0.0024347
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006699, 0.0006764
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006080, 0.0006140
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022691, 0.0022913
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017833, 0.0017660
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001524, 0.0001539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015887, 0.0015439
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004479, 0.0004353
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033048, 0.0032116
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004373, 0.0004250
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024002, 0.0024698
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006668, 0.0006862
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006053, 0.0006228
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022588, 0.0023243
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018090, 0.0017580
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001517, 0.0001561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015687, 0.0015478
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004423, 0.0004364
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032633, 0.0032197
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004318, 0.0004261
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024062, 0.0024388
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006685, 0.0006776
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006068, 0.0006150
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022645, 0.0022952
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017863, 0.0017624
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001521, 0.0001541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015913, 0.0015406
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004486, 0.0004343
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033102, 0.0032047
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004381, 0.0004241
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023950, 0.0024739
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006654, 0.0006873
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006040, 0.0006239
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022539, 0.0023282
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018120, 0.0017542
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001513, 0.0001563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015588, 0.0015519
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004395, 0.0004376
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032427, 0.0032284
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004291, 0.0004272
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024127, 0.0024234
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006703, 0.0006733
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006084, 0.0006111
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022706, 0.0022806
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017750, 0.0017672
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001525, 0.0001531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015814, 0.0015444
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004459, 0.0004354
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032896, 0.0032126
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004353, 0.0004251
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024009, 0.0024584
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006670, 0.0006830
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006055, 0.0006200
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022595, 0.0023137
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018007, 0.0017586
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001517, 0.0001554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015615, 0.0015490
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004402, 0.0004367
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032481, 0.0032221
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004298, 0.0004264
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024080, 0.0024275
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006690, 0.0006744
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006073, 0.0006122
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022662, 0.0022845
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017780, 0.0017638
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001522, 0.0001534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015840, 0.0015416
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004466, 0.0004346
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032951, 0.0032069
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004361, 0.0004244
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023967, 0.0024626
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006659, 0.0006842
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006044, 0.0006210
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022555, 0.0023175
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018037, 0.0017555
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001515, 0.0001556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015658, 0.0015476
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004415, 0.0004363
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032572, 0.0032193
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004310, 0.0004260
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024059, 0.0024342
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006684, 0.0006763
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006067, 0.0006139
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022642, 0.0022909
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017830, 0.0017623
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001520, 0.0001538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015884, 0.0015405
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004478, 0.0004343
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033042, 0.0032046
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004373, 0.0004241
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023949, 0.0024693
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006654, 0.0006861
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006040, 0.0006227
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022539, 0.0023239
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018087, 0.0017542
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001513, 0.0001560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015687, 0.0015446
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004423, 0.0004355
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032631, 0.0032130
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004318, 0.0004252
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024012, 0.0024386
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006671, 0.0006775
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006055, 0.0006150
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022598, 0.0022950
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017862, 0.0017588
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001517, 0.0001541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015912, 0.0015372
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004486, 0.0004334
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0033101, 0.0031977
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004380, 0.0004232
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023898, 0.0024737
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006640, 0.0006873
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006027, 0.0006238
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022491, 0.0023281
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018119, 0.0017504
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001510, 0.0001563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015599, 0.0015504
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004398, 0.0004371
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032448, 0.0032252
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004294, 0.0004268
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024103, 0.0024250
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006697, 0.0006737
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006079, 0.0006115
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022684, 0.0022822
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017762, 0.0017655
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001523, 0.0001532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015824, 0.0015432
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004461, 0.0004351
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032918, 0.0032102
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004356, 0.0004248
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023991, 0.0024601
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006666, 0.0006835
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006050, 0.0006204
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022579, 0.0023152
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018019, 0.0017573
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001516, 0.0001555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015625, 0.0015470
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004405, 0.0004362
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032504, 0.0032180
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004301, 0.0004259
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024050, 0.0024291
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006682, 0.0006749
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006065, 0.0006126
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022633, 0.0022861
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017793, 0.0017616
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001520, 0.0001535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015851, 0.0015401
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004469, 0.0004342
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032973, 0.0032038
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004364, 0.0004240
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023943, 0.0024642
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006652, 0.0006846
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006038, 0.0006214
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022533, 0.0023191
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0018050, 0.0017538
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001513, 0.0001557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015461, 0.0015654
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004359, 0.0004413
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032162, 0.0032564
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004256, 0.0004309
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024336, 0.0024036
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006761, 0.0006678
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006137, 0.0006062
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022903, 0.0022621
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017606, 0.0017825
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001538, 0.0001519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015687, 0.0015566
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004423, 0.0004388
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032632, 0.0032379
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004318, 0.0004285
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024198, 0.0024387
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006723, 0.0006775
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006102, 0.0006150
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022773, 0.0022951
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017863, 0.0017725
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001529, 0.0001541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015492, 0.0015627
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004368, 0.0004406
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032227, 0.0032508
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004265, 0.0004302
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024295, 0.0024084
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006750, 0.0006691
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006127, 0.0006074
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022864, 0.0022666
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017641, 0.0017795
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001535, 0.0001522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015718, 0.0015539
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004431, 0.0004381
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032696, 0.0032324
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004327, 0.0004278
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024157, 0.0024435
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006711, 0.0006789
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006092, 0.0006162
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022734, 0.0022996
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017898, 0.0017694
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001527, 0.0001544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015432, 0.0015693
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004351, 0.0004425
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032102, 0.0032645
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004248, 0.0004320
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024397, 0.0023991
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006778, 0.0006665
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006153, 0.0006050
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022960, 0.0022578
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017573, 0.0017870
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001542, 0.0001516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015658, 0.0015627
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004415, 0.0004406
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032571, 0.0032507
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004310, 0.0004302
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024293, 0.0024342
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006749, 0.0006763
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006126, 0.0006139
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022863, 0.0022908
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017830, 0.0017794
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001535, 0.0001538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015465, 0.0015663
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004360, 0.0004416
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032170, 0.0032581
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004257, 0.0004312
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024349, 0.0024042
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006765, 0.0006680
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006141, 0.0006063
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022915, 0.0022626
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017610, 0.0017835
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001539, 0.0001519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015691, 0.0015598
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004424, 0.0004398
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032640, 0.0032448
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004319, 0.0004294
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024249, 0.0024393
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006737, 0.0006777
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006115, 0.0006152
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022821, 0.0022956
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017867, 0.0017762
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001532, 0.0001541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015476, 0.0015634
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004363, 0.0004408
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032194, 0.0032521
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004260, 0.0004304
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024304, 0.0024060
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006752, 0.0006684
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006129, 0.0006067
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022873, 0.0022643
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017623, 0.0017802
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001536, 0.0001520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015702, 0.0015555
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004427, 0.0004385
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032663, 0.0032357
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004322, 0.0004282
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024182, 0.0024411
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006718, 0.0006782
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006098, 0.0006156
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022758, 0.0022973
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017880, 0.0017712
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001528, 0.0001543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015504, 0.0015608
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004371, 0.0004401
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032251, 0.0032468
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004268, 0.0004297
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024265, 0.0024102
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006741, 0.0006696
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006119, 0.0006078
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022836, 0.0022683
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017654, 0.0017773
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001533, 0.0001523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015729, 0.0015528
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004435, 0.0004378
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032720, 0.0032302
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004330, 0.0004275
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024141, 0.0024453
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006707, 0.0006794
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006088, 0.0006167
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022719, 0.0023013
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017911, 0.0017682
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001526, 0.0001545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015465, 0.0015698
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004360, 0.0004426
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032171, 0.0032656
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004257, 0.0004321
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024405, 0.0024043
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006780, 0.0006680
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006155, 0.0006063
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022968, 0.0022627
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017611, 0.0017876
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001542, 0.0001519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015691, 0.0015627
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004424, 0.0004406
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032641, 0.0032508
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004319, 0.0004302
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024295, 0.0024394
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006750, 0.0006777
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006127, 0.0006152
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022864, 0.0022957
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017868, 0.0017795
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001535, 0.0001542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015499, 0.0015669
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004370, 0.0004418
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032241, 0.0032594
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004267, 0.0004313
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024359, 0.0024095
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006768, 0.0006694
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006143, 0.0006076
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022924, 0.0022676
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017649, 0.0017842
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001539, 0.0001523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015724, 0.0015601
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004433, 0.0004399
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0032710, 0.0032454
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004329, 0.0004295
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0024254, 0.0024446
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006738, 0.0006792
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0006116, 0.0006165
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022826, 0.0023006
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017906, 0.0017765
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001533, 0.0001545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
time: 0.83 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003265, upper bound: 0.0003323
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003262, upper bound: 0.0003324
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003246, upper bound: 0.0003333
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003335
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003273, upper bound: 0.0003303
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003272, upper bound: 0.0003308
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003257, upper bound: 0.0003311
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003318
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003256
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003311, upper bound: 0.0003258
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003308, upper bound: 0.0003273
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003303, upper bound: 0.0003273
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003335, upper bound: 0.0003244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003333, upper bound: 0.0003246
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003324, upper bound: 0.0003262
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0003323, upper bound: 0.0003265

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015033, 0.0015037
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004238, 0.0004239
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031273, 0.0031280
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004138, 0.0004139
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023377, 0.0023371
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006495, 0.0006493
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005895, 0.0005894
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0022000, 0.0021995
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017119, 0.0017123
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001477, 0.0001477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003222, upper bound: 0.0003192
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003127, upper bound: 0.0003278
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015100, 0.0014984
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004257, 0.0004224
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031412, 0.0031169
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004157, 0.0004125
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023294, 0.0023475
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006472, 0.0006522
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005874, 0.0005920
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021922, 0.0022093
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017195, 0.0017062
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001472, 0.0001484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003218, upper bound: 0.0003192
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003126, upper bound: 0.0003279
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015259, 0.0014979
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004302, 0.0004223
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031741, 0.0031159
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004200, 0.0004123
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023287, 0.0023721
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006470, 0.0006590
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005873, 0.0005982
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021915, 0.0022324
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017375, 0.0017057
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001472, 0.0001499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003222, upper bound: 0.0003192
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003127, upper bound: 0.0003278
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015326, 0.0014914
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004321, 0.0004205
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031881, 0.0031023
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004219, 0.0004105
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023185, 0.0023826
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006441, 0.0006619
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005847, 0.0006008
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021820, 0.0022423
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017452, 0.0016982
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001465, 0.0001506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003218, upper bound: 0.0003192
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003126, upper bound: 0.0003279
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015060, 0.0015004
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004246, 0.0004230
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031327, 0.0031212
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004146, 0.0004130
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023326, 0.0023412
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006481, 0.0006505
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005882, 0.0005904
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021952, 0.0022033
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017149, 0.0017086
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001474, 0.0001479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003202, upper bound: 0.0003206
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003105, upper bound: 0.0003288
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015129, 0.0014949
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004266, 0.0004215
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031472, 0.0031098
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004165, 0.0004115
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023241, 0.0023521
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006457, 0.0006535
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005861, 0.0005932
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021872, 0.0022135
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017228, 0.0017023
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001469, 0.0001486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003200, upper bound: 0.0003205
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003105, upper bound: 0.0003291
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015285, 0.0014948
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004309, 0.0004214
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031796, 0.0031095
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004208, 0.0004115
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023238, 0.0023762
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006456, 0.0006602
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005860, 0.0005992
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021870, 0.0022363
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017405, 0.0017021
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001469, 0.0001502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003202, upper bound: 0.0003206
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003105, upper bound: 0.0003288
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015355, 0.0014880
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004329, 0.0004195
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031941, 0.0030953
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004227, 0.0004096
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023132, 0.0023871
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006427, 0.0006632
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005834, 0.0006020
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021770, 0.0022465
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017485, 0.0016944
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001462, 0.0001508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003200, upper bound: 0.0003205
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003105, upper bound: 0.0003291
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0014961, 0.0015031
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004218, 0.0004238
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031121, 0.0031267
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004118, 0.0004138
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023367, 0.0023258
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006492, 0.0006462
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005893, 0.0005865
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021991, 0.0021888
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017036, 0.0017116
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001477, 0.0001470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003229, upper bound: 0.0003165
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003143, upper bound: 0.0003258
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015031, 0.0014979
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004238, 0.0004223
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031268, 0.0031159
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004138, 0.0004123
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023286, 0.0023368
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006470, 0.0006492
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005872, 0.0005893
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021915, 0.0021992
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017116, 0.0017056
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001472, 0.0001477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003228, upper bound: 0.0003165
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003144, upper bound: 0.0003263
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015186, 0.0014970
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004281, 0.0004221
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031590, 0.0031141
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004180, 0.0004121
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023273, 0.0023608
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006466, 0.0006559
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005869, 0.0005954
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021902, 0.0022218
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017292, 0.0017046
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001471, 0.0001492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003229, upper bound: 0.0003165
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003143, upper bound: 0.0003258
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015257, 0.0014912
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004301, 0.0004204
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031737, 0.0031019
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004200, 0.0004105
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023182, 0.0023718
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006441, 0.0006590
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005846, 0.0005981
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021817, 0.0022321
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017373, 0.0016980
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001465, 0.0001499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003228, upper bound: 0.0003165
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003144, upper bound: 0.0003263
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0014987, 0.0015000
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004225, 0.0004229
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031176, 0.0031204
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004126, 0.0004129
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023320, 0.0023299
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006479, 0.0006473
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005881, 0.0005876
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021947, 0.0021927
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017066, 0.0017081
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001474, 0.0001472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003213, upper bound: 0.0003179
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003127, upper bound: 0.0003266
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0090131, -0.0060574, -0.0090131, -0.0060574, -0.0015058, 0.0014949
1: -0.0054798, -0.0046465, -0.0054798, -0.0046465, -0.0004245, 0.0004215
2: -0.0018712, 0.0042773, -0.0018712, 0.0042773, -0.0031323, 0.0031098
3: 0.0013797, 0.0021933, 0.0013797, 0.0021933, -0.0004145, 0.0004115
4: 0.0028953, 0.0074903, 0.0028953, 0.0074903, -0.0023241, 0.0023409
5: 0.9963107, 0.9975873, 0.9963107, 0.9975873, -0.0006457, 0.0006504
6: 0.0045348, 0.0056936, 0.0045348, 0.0056936, -0.0005861, 0.0005903
7: -0.0064584, -0.0021339, -0.0064584, -0.0021339, -0.0021872, 0.0022030
8: -0.0075320, -0.0041663, -0.0075320, -0.0041663, -0.0017146, 0.0017023
9: -0.0036503, -0.0033599, -0.0036503, -0.0033599, -0.0001469, 0.0001479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.20 + 597.19 = 600.38 seconds
