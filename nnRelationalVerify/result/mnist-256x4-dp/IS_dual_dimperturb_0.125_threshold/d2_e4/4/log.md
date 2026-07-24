## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00058656


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0037835, 0.0045893, 0.0037835, 0.0045893, -0.0005874, 0.0005874)
1: (0.0018689, 0.0019853, 0.0018689, 0.0019853, -0.0000849, 0.0000849)
2: (0.0118226, 0.0122681, 0.0118226, 0.0122681, -0.0003248, 0.0003248)
3: (-0.0024530, -0.0019922, -0.0024530, -0.0019922, -0.0003359, 0.0003359)
4: (-0.0018802, -0.0013814, -0.0018802, -0.0013814, -0.0003636, 0.0003636)
5: (0.0054206, 0.0058927, 0.0054206, 0.0058927, -0.0003441, 0.0003441)
6: (-0.0007928, 0.0010801, -0.0007928, 0.0010801, -0.0013653, 0.0013653)
7: (-0.0040278, -0.0014770, -0.0040278, -0.0014770, -0.0018594, 0.0018594)
8: (0.9863766, 0.9881734, 0.9863766, 0.9881734, -0.0013098, 0.0013098)
9: (-0.0051519, -0.0035209, -0.0051519, -0.0035209, -0.0011890, 0.0011890)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.57 + 1.30 = 2.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0006522, upper bound: 0.0006522

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0006098
time: 0.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006100, upper bound: 0.0006100
time: 0.46 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0006098
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 8, lower bound: -0.0006100, upper bound: 0.0006100

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0038821, 0.0045863, 0.0038218, 0.0045882, -0.0004874, 0.0005452
1: 0.0018831, 0.0019849, 0.0018744, 0.0019852, -0.0000704, 0.0000788
2: 0.0118242, 0.0122136, 0.0118232, 0.0122469, -0.0003014, 0.0002695
3: -0.0024513, -0.0020486, -0.0024524, -0.0020141, -0.0003117, 0.0002787
4: -0.0018192, -0.0013833, -0.0018565, -0.0013821, -0.0003017, 0.0003375
5: 0.0054224, 0.0058349, 0.0054213, 0.0058703, -0.0003194, 0.0002855
6: -0.0007859, 0.0008510, -0.0007902, 0.0009911, -0.0012671, 0.0011329
7: -0.0037157, -0.0014864, -0.0039065, -0.0014805, -0.0015430, 0.0017257
8: 0.9865964, 0.9881668, 0.9864621, 0.9881710, -0.0010869, 0.0012156
9: -0.0051459, -0.0037204, -0.0051497, -0.0035984, -0.0011034, 0.0009866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005198, upper bound: 0.0005514
time: 0.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005198, upper bound: 0.0005798
time: 0.47 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0038264, 0.0046568, 0.0038009, 0.0045885, -0.0005104, 0.0006712
1: 0.0018751, 0.0019951, 0.0018714, 0.0019852, -0.0000737, 0.0000970
2: 0.0117852, 0.0122443, 0.0118230, 0.0122585, -0.0003711, 0.0002822
3: -0.0024916, -0.0020168, -0.0024526, -0.0020022, -0.0003838, 0.0002918
4: -0.0018537, -0.0013397, -0.0018695, -0.0013819, -0.0003159, 0.0004155
5: 0.0053811, 0.0058675, 0.0054211, 0.0058825, -0.0003932, 0.0002990
6: -0.0009497, 0.0009803, -0.0007910, 0.0010397, -0.0015601, 0.0011863
7: -0.0038918, -0.0012634, -0.0039727, -0.0014795, -0.0016156, 0.0021248
8: 0.9864724, 0.9883239, 0.9864153, 0.9881716, -0.0011381, 0.0014967
9: -0.0052885, -0.0036078, -0.0051503, -0.0035561, -0.0013586, 0.0010331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005800, upper bound: 0.0005516
time: 0.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005800, upper bound: 0.0005800
time: 0.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.49 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.49
Output dim: 8, lower bound: -0.0005198, upper bound: 0.0005514
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.49
Output dim: 8, lower bound: -0.0005198, upper bound: 0.0005798
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 2.49
Output dim: 8, lower bound: -0.0005800, upper bound: 0.0005516
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 2.49
Output dim: 8, lower bound: -0.0005800, upper bound: 0.0005800

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.87 + 6.18 = 9.05 seconds
