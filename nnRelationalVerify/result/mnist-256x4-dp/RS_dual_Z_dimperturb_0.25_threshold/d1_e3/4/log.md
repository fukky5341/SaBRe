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
Threshold: 0.00104286


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005471, 0.0005471)
1: (-0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0004046, 0.0004046)
2: (0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001747, 0.0001747)
3: (0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0003002, 0.0003002)
4: (-0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003442, 0.0003442)
5: (-0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002484, 0.0002484)
6: (-0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0006582, 0.0006582)
7: (-0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0019865, 0.0019865)
8: (0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0018516, 0.0018516)
9: (0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0013076, 0.0013076)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.32 = 2.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0014219, upper bound: 0.0014219

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013479, upper bound: 0.0013479
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013479, upper bound: 0.0013479
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 8, lower bound: -0.0013479, upper bound: 0.0013479
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 8, lower bound: -0.0013479, upper bound: 0.0013479

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005395, 0.0005434
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003990, 0.0004016
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001735, 0.0001723
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002995, 0.0002980
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003406, 0.0003416
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002469, 0.0002449
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0006580, 0.0006580
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0019657, 0.0019704
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0018335, 0.0018346
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0012965, 0.0012942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012752, upper bound: 0.0012915
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012915, upper bound: 0.0012752
time: 0.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005471, 0.0005395
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0004046, 0.0003990
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001723, 0.0001747
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002980, 0.0003002
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003416, 0.0003442
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002449, 0.0002484
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0006582, 0.0006580
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0019704, 0.0019865
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0018346, 0.0018516
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0013076, 0.0012965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012752, upper bound: 0.0012915
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012915, upper bound: 0.0012752
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 8, lower bound: -0.0012752, upper bound: 0.0012915
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 8, lower bound: -0.0012915, upper bound: 0.0012752
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 8, lower bound: -0.0012752, upper bound: 0.0012915
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 8, lower bound: -0.0012915, upper bound: 0.0012752

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005272, 0.0005289
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003943, 0.0003960
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001681, 0.0001678
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002882, 0.0002862
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003110, 0.0003156
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002446, 0.0002430
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0006048, 0.0006016
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017989, 0.0018238
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016968, 0.0017091
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0012030, 0.0011872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012678, upper bound: 0.0012614
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012562, upper bound: 0.0012842
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005251, 0.0005299
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003934, 0.0003968
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001684, 0.0001670
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002878, 0.0002864
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003152, 0.0003120
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002448, 0.0002427
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0006017, 0.0006047
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0018212, 0.0018036
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0017097, 0.0016979
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011895, 0.0012017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012842, upper bound: 0.0012562
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012614, upper bound: 0.0012678
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005347, 0.0005251
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003999, 0.0003934
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001670, 0.0001701
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002864, 0.0002886
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003120, 0.0003182
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002427, 0.0002464
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0006049, 0.0006017
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0018036, 0.0018399
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016979, 0.0017261
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0012141, 0.0011895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012678, upper bound: 0.0012614
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012562, upper bound: 0.0012842
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005326, 0.0005272
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003990, 0.0003943
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001678, 0.0001693
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002862, 0.0002888
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003156, 0.0003147
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002430, 0.0002462
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0006018, 0.0006048
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0018238, 0.0018197
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0017091, 0.0017149
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0012006, 0.0012030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012842, upper bound: 0.0012562
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012614, upper bound: 0.0012678
time: 0.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 8, lower bound: -0.0012678, upper bound: 0.0012614
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 8, lower bound: -0.0012562, upper bound: 0.0012842
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 8, lower bound: -0.0012842, upper bound: 0.0012562
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 8, lower bound: -0.0012614, upper bound: 0.0012678
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 8, lower bound: -0.0012678, upper bound: 0.0012614
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 8, lower bound: -0.0012562, upper bound: 0.0012842
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 8, lower bound: -0.0012842, upper bound: 0.0012562
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 8, lower bound: -0.0012614, upper bound: 0.0012678

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005216, 0.0005234
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003928, 0.0003946
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001659, 0.0001656
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002771, 0.0002778
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002872, 0.0002864
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002443, 0.0002427
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005363, 0.0005476
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016660, 0.0016640
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016031, 0.0016060
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011035, 0.0011039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012569, upper bound: 0.0011125
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011358, upper bound: 0.0012512
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005212, 0.0005233
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003928, 0.0003946
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001659, 0.0001654
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002798, 0.0002752
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002819, 0.0002930
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002443, 0.0002426
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005531, 0.0005331
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016390, 0.0016987
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0015937, 0.0016249
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011239, 0.0010876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012459, upper bound: 0.0011557
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011066, upper bound: 0.0012735
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005195, 0.0005245
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003919, 0.0003954
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001663, 0.0001648
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002767, 0.0002783
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002923, 0.0002829
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002445, 0.0002424
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005332, 0.0005528
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016948, 0.0016437
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016247, 0.0015948
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0010899, 0.0011219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012735, upper bound: 0.0011066
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011557, upper bound: 0.0012459
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005191, 0.0005243
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003918, 0.0003953
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001662, 0.0001647
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002791, 0.0002753
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002860, 0.0002884
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002445, 0.0002423
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005479, 0.0005363
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016613, 0.0016735
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016066, 0.0016080
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011081, 0.0011021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012512, upper bound: 0.0011358
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011125, upper bound: 0.0012569
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005288, 0.0005191
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003984, 0.0003918
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001647, 0.0001678
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002753, 0.0002797
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002884, 0.0002889
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002423, 0.0002461
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005364, 0.0005479
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016735, 0.0016791
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016080, 0.0016230
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011139, 0.0011081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012569, upper bound: 0.0011125
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011358, upper bound: 0.0012512
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005285, 0.0005195
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003984, 0.0003919
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001648, 0.0001677
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002783, 0.0002773
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002829, 0.0002957
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002424, 0.0002461
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005536, 0.0005332
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016437, 0.0017149
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0015948, 0.0016428
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011351, 0.0010899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012459, upper bound: 0.0011557
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011066, upper bound: 0.0012735
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005267, 0.0005212
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003975, 0.0003928
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001654, 0.0001670
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002752, 0.0002802
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002930, 0.0002854
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002426, 0.0002458
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005333, 0.0005531
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016987, 0.0016588
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016249, 0.0016118
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011003, 0.0011239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012735, upper bound: 0.0011066
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011557, upper bound: 0.0012459
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005264, 0.0005216
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003974, 0.0003928
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001656, 0.0001670
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002778, 0.0002774
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002864, 0.0002911
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002427, 0.0002458
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005484, 0.0005363
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016640, 0.0016897
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016060, 0.0016259
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011192, 0.0011035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012512, upper bound: 0.0011358
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011125, upper bound: 0.0012569
time: 0.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0012569, upper bound: 0.0011125
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0011358, upper bound: 0.0012512
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0012459, upper bound: 0.0011557
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0011066, upper bound: 0.0012735
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0012735, upper bound: 0.0011066
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0011557, upper bound: 0.0012459
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0012512, upper bound: 0.0011358
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0011125, upper bound: 0.0012569
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0012569, upper bound: 0.0011125
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0011358, upper bound: 0.0012512
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0012459, upper bound: 0.0011557
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0011066, upper bound: 0.0012735
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0012735, upper bound: 0.0011066
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0011557, upper bound: 0.0012459
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0012512, upper bound: 0.0011358
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 8, lower bound: -0.0011125, upper bound: 0.0012569

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005288, 0.0005313
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003971, 0.0003973
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001689, 0.0001682
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002783, 0.0002876
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002969, 0.0002955
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002452, 0.0002453
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005447, 0.0005658
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017197, 0.0017170
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016612, 0.0016622
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011399, 0.0011400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008358, upper bound: 0.0007634
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008360, upper bound: 0.0007634
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005296, 0.0005305
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003955, 0.0003990
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001687, 0.0001686
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002891, 0.0002764
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002909, 0.0003013
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002468, 0.0002435
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005709, 0.0005415
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016920, 0.0017449
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016499, 0.0016737
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011559, 0.0011240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007819, upper bound: 0.0008219
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007819, upper bound: 0.0008219
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005288, 0.0005313
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003971, 0.0003973
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001689, 0.0001682
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002783, 0.0002876
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002969, 0.0002955
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002452, 0.0002453
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005447, 0.0005658
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017197, 0.0017170
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016612, 0.0016622
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011399, 0.0011400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008081, upper bound: 0.0008099
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008081, upper bound: 0.0008099
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005296, 0.0005305
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003955, 0.0003990
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001687, 0.0001686
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002891, 0.0002764
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002909, 0.0003013
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002468, 0.0002435
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005709, 0.0005415
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016920, 0.0017449
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016499, 0.0016737
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011559, 0.0011240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007457, upper bound: 0.0008572
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007457, upper bound: 0.0008566
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005272, 0.0005323
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003966, 0.0003981
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001692, 0.0001676
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002779, 0.0002879
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003007, 0.0002919
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002454, 0.0002453
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005416, 0.0005697
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017424, 0.0016967
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016745, 0.0016510
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011263, 0.0011546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008566, upper bound: 0.0007457
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008572, upper bound: 0.0007457
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005275, 0.0005317
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003947, 0.0003993
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001690, 0.0001678
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002888, 0.0002765
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002951, 0.0002981
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002469, 0.0002433
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005669, 0.0005446
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017143, 0.0017266
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016628, 0.0016628
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011430, 0.0011386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008099, upper bound: 0.0008081
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008099, upper bound: 0.0008081
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005272, 0.0005323
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003966, 0.0003981
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001692, 0.0001676
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002779, 0.0002879
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003007, 0.0002919
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002454, 0.0002453
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005416, 0.0005697
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017424, 0.0016967
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016745, 0.0016510
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011263, 0.0011546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008219, upper bound: 0.0007819
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008219, upper bound: 0.0007819
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005275, 0.0005317
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003947, 0.0003993
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001690, 0.0001678
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002888, 0.0002765
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002951, 0.0002981
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002469, 0.0002433
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005669, 0.0005446
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017143, 0.0017266
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016628, 0.0016628
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011430, 0.0011386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007634, upper bound: 0.0008360
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007634, upper bound: 0.0008358
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005353, 0.0005275
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0004016, 0.0003947
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001678, 0.0001703
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002765, 0.0002888
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002981, 0.0002980
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002433, 0.0002480
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005448, 0.0005669
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017266, 0.0017328
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016628, 0.0016803
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011508, 0.0011430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008358, upper bound: 0.0007634
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008360, upper bound: 0.0007634
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005365, 0.0005272
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0004005, 0.0003966
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001676, 0.0001708
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002879, 0.0002780
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002919, 0.0003042
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002453, 0.0002465
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005722, 0.0005416
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016967, 0.0017625
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016510, 0.0016927
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011679, 0.0011263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007819, upper bound: 0.0008219
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007819, upper bound: 0.0008219
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005353, 0.0005275
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0004016, 0.0003947
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001678, 0.0001703
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002765, 0.0002888
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002981, 0.0002980
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002433, 0.0002480
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005448, 0.0005669
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017266, 0.0017328
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016628, 0.0016803
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011508, 0.0011430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008081, upper bound: 0.0008099
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008081, upper bound: 0.0008099
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005365, 0.0005272
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0004005, 0.0003966
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001676, 0.0001708
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002879, 0.0002780
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002919, 0.0003042
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002453, 0.0002465
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005722, 0.0005416
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0016967, 0.0017625
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016510, 0.0016927
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011679, 0.0011263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007457, upper bound: 0.0008572
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007457, upper bound: 0.0008566
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005336, 0.0005296
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0004011, 0.0003955
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001686, 0.0001697
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002764, 0.0002891
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003013, 0.0002945
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002435, 0.0002479
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005417, 0.0005709
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017449, 0.0017126
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016737, 0.0016692
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011372, 0.0011559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008566, upper bound: 0.0007457
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008572, upper bound: 0.0007457
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005344, 0.0005288
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003996, 0.0003971
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001682, 0.0001700
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002876, 0.0002782
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002955, 0.0003010
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002453, 0.0002462
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005682, 0.0005447
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017170, 0.0017442
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016622, 0.0016818
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011550, 0.0011399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008099, upper bound: 0.0008081
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008099, upper bound: 0.0008081
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005336, 0.0005296
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0004011, 0.0003955
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001686, 0.0001697
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002764, 0.0002891
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0003013, 0.0002945
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002435, 0.0002479
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005417, 0.0005709
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017449, 0.0017126
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016737, 0.0016692
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011372, 0.0011559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008219, upper bound: 0.0007819
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008219, upper bound: 0.0007819
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0167001, 0.0176139, 0.0167001, 0.0176139, -0.0005344, 0.0005288
1: -0.0007591, -0.0000971, -0.0007591, -0.0000971, -0.0003996, 0.0003971
2: 0.0037765, 0.0040716, 0.0037765, 0.0040716, -0.0001682, 0.0001700
3: 0.0016429, 0.0022382, 0.0016429, 0.0022382, -0.0002876, 0.0002782
4: -0.0041623, -0.0034378, -0.0041623, -0.0034378, -0.0002955, 0.0003010
5: -0.0000933, 0.0003127, -0.0000933, 0.0003127, -0.0002453, 0.0002462
6: -0.0041120, -0.0026591, -0.0041120, -0.0026591, -0.0005682, 0.0005447
7: -0.0201327, -0.0159880, -0.0201327, -0.0159880, -0.0017170, 0.0017442
8: 0.9769187, 0.9805782, 0.9769187, 0.9805782, -0.0016622, 0.0016818
9: 0.0028097, 0.0055113, 0.0028097, 0.0055113, -0.0011550, 0.0011399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 99
type: RSZ, layer: 3, pos: 115
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Candidate
type: RSZ, layer: 3, pos: 99

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 115

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007634, upper bound: 0.0008360
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007634, upper bound: 0.0008358
time: 0.52 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008358, upper bound: 0.0007634
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008360, upper bound: 0.0007634
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007819, upper bound: 0.0008219
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007819, upper bound: 0.0008219
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008081, upper bound: 0.0008099
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008081, upper bound: 0.0008099
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007457, upper bound: 0.0008572
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007457, upper bound: 0.0008566
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008566, upper bound: 0.0007457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008572, upper bound: 0.0007457
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008099, upper bound: 0.0008081
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008099, upper bound: 0.0008081
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008219, upper bound: 0.0007819
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008219, upper bound: 0.0007819
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007634, upper bound: 0.0008360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007634, upper bound: 0.0008358
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008358, upper bound: 0.0007634
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008360, upper bound: 0.0007634
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007819, upper bound: 0.0008219
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007819, upper bound: 0.0008219
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008081, upper bound: 0.0008099
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008081, upper bound: 0.0008099
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007457, upper bound: 0.0008572
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007457, upper bound: 0.0008566
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008566, upper bound: 0.0007457
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008572, upper bound: 0.0007457
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008099, upper bound: 0.0008081
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008099, upper bound: 0.0008081
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008219, upper bound: 0.0007819
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0008219, upper bound: 0.0007819
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007634, upper bound: 0.0008360
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0007634, upper bound: 0.0008358

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.61 + 100.96 = 103.57 seconds
