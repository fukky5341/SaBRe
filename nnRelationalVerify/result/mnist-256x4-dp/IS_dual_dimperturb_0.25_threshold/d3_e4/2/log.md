## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0024309


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040167, -0.0036151, -0.0040167, -0.0036151, -0.0002935, 0.0002935)
1: (-0.0002352, 0.0019882, -0.0002352, 0.0019882, -0.0016252, 0.0016252)
2: (0.0105244, 0.0154916, 0.0105244, 0.0154916, -0.0036309, 0.0036309)
3: (0.0008062, 0.0028993, 0.0008062, 0.0028993, -0.0015301, 0.0015301)
4: (0.9998780, 1.0079987, 0.9998780, 1.0079987, -0.0059360, 0.0059360)
5: (0.0022177, 0.0037975, 0.0022177, 0.0037975, -0.0011548, 0.0011548)
6: (-0.0106848, -0.0086289, -0.0106848, -0.0086289, -0.0015028, 0.0015028)
7: (-0.0101663, -0.0099041, -0.0101663, -0.0099041, -0.0001917, 0.0001917)
8: (-0.0048922, -0.0034718, -0.0048922, -0.0034718, -0.0010383, 0.0010383)
9: (-0.0007904, 0.0063207, -0.0007904, 0.0063207, -0.0051981, 0.0051981)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.80 + 1.58 = 3.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0040029, upper bound: 0.0040029

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0034426, upper bound: 0.0036949
time: 0.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0036949, upper bound: 0.0036949
time: 0.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 4, lower bound: -0.0034426, upper bound: 0.0036949
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 4, lower bound: -0.0036949, upper bound: 0.0036949

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0040165, -0.0036444, -0.0040166, -0.0036173, -0.0002907, 0.0002604
1: -0.0000733, 0.0019874, -0.0002231, 0.0019881, -0.0014419, 0.0016096
2: 0.0105261, 0.0151298, 0.0105245, 0.0154646, -0.0035960, 0.0032213
3: 0.0009586, 0.0028986, 0.0008175, 0.0028993, -0.0013575, 0.0015154
4: 1.0004693, 1.0079958, 0.9999220, 1.0079985, -0.0052664, 0.0058791
5: 0.0023327, 0.0037969, 0.0022262, 0.0037974, -0.0010245, 0.0011437
6: -0.0106841, -0.0087786, -0.0106847, -0.0086401, -0.0014884, 0.0013333
7: -0.0101662, -0.0099232, -0.0101663, -0.0099055, -0.0001899, 0.0001701
8: -0.0047888, -0.0034723, -0.0048845, -0.0034718, -0.0009212, 0.0010284
9: -0.0007879, 0.0058029, -0.0007902, 0.0062821, -0.0051482, 0.0046117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0034411, upper bound: 0.0034411
time: 0.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0034411, upper bound: 0.0036949
time: 0.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0036558, -0.0040165, -0.0036343, -0.0003898, 0.0002752
1: -0.0000100, 0.0024358, -0.0001290, 0.0019874, -0.0015240, 0.0021582
2: 0.0095244, 0.0149885, 0.0105260, 0.0152543, -0.0048217, 0.0034049
3: 0.0010182, 0.0033208, 0.0009062, 0.0028987, -0.0014348, 0.0020319
4: 1.0007004, 1.0096335, 1.0002658, 1.0079960, -0.0055666, 0.0078828
5: 0.0023777, 0.0041155, 0.0022931, 0.0037969, -0.0010829, 0.0015335
6: -0.0110987, -0.0088371, -0.0106841, -0.0087271, -0.0019957, 0.0014093
7: -0.0102191, -0.0099306, -0.0101662, -0.0099166, -0.0002546, 0.0001798
8: -0.0047484, -0.0031858, -0.0048244, -0.0034722, -0.0009737, 0.0013788
9: -0.0022221, 0.0056005, -0.0007880, 0.0059811, -0.0069028, 0.0048745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0033085
time: 0.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0033319
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.31 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 4, lower bound: -0.0034411, upper bound: 0.0034411
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 4, lower bound: -0.0034411, upper bound: 0.0036949
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0033085
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0033319

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040165, -0.0036444, -0.0040165, -0.0036444, -0.0002602, 0.0002602
1: -0.0000733, 0.0019874, -0.0000733, 0.0019874, -0.0014409, 0.0014409
2: 0.0105261, 0.0151298, 0.0105261, 0.0151298, -0.0032192, 0.0032192
3: 0.0009586, 0.0028986, 0.0009586, 0.0028986, -0.0013566, 0.0013566
4: 1.0004693, 1.0079958, 1.0004693, 1.0079958, -0.0052630, 0.0052630
5: 0.0023327, 0.0037969, 0.0023327, 0.0037969, -0.0010239, 0.0010239
6: -0.0106841, -0.0087786, -0.0106841, -0.0087786, -0.0013324, 0.0013324
7: -0.0101662, -0.0099232, -0.0101662, -0.0099232, -0.0001700, 0.0001700
8: -0.0047888, -0.0034723, -0.0047888, -0.0034723, -0.0009206, 0.0009206
9: -0.0007879, 0.0058029, -0.0007879, 0.0058029, -0.0046087, 0.0046087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030850, upper bound: 0.0031689
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031583, upper bound: 0.0031689
time: 0.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040165, -0.0036444, -0.0040975, -0.0036558, -0.0002745, 0.0003676
1: -0.0000733, 0.0019874, -0.0000100, 0.0024358, -0.0020351, 0.0015199
2: 0.0105261, 0.0151298, 0.0095244, 0.0149885, -0.0033956, 0.0045467
3: 0.0009586, 0.0028986, 0.0010182, 0.0033208, -0.0019160, 0.0014309
4: 1.0004693, 1.0079958, 1.0007004, 1.0096335, -0.0074334, 0.0055513
5: 0.0023327, 0.0037969, 0.0023777, 0.0041155, -0.0014461, 0.0010800
6: -0.0106841, -0.0087786, -0.0110987, -0.0088371, -0.0014054, 0.0018819
7: -0.0101662, -0.0099232, -0.0102191, -0.0099306, -0.0001793, 0.0002401
8: -0.0047888, -0.0034723, -0.0047484, -0.0031858, -0.0013002, 0.0009710
9: -0.0007879, 0.0058029, -0.0022221, 0.0056005, -0.0048612, 0.0065092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030850, upper bound: 0.0033319
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031583, upper bound: 0.0033319
time: 0.61 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0036558, -0.0040165, -0.0036467, -0.0003750, 0.0002752
1: -0.0000100, 0.0024358, -0.0000602, 0.0019871, -0.0015236, 0.0020765
2: 0.0095244, 0.0149885, 0.0105268, 0.0151005, -0.0046390, 0.0034040
3: 0.0010182, 0.0033208, 0.0009709, 0.0028983, -0.0014344, 0.0019549
4: 1.0007004, 1.0096335, 1.0005172, 1.0079947, -0.0055651, 0.0075843
5: 0.0023777, 0.0041155, 0.0023420, 0.0037967, -0.0010826, 0.0014754
6: -0.0110987, -0.0088371, -0.0106838, -0.0087907, -0.0019201, 0.0014089
7: -0.0102191, -0.0099306, -0.0101662, -0.0099247, -0.0002449, 0.0001797
8: -0.0047484, -0.0031858, -0.0047804, -0.0034725, -0.0009734, 0.0013266
9: -0.0022221, 0.0056005, -0.0007869, 0.0057610, -0.0066413, 0.0048732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0030840
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0030840
time: 0.66 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0036712, -0.0040739, -0.0036637, -0.0003839, 0.0003402
1: 0.0000752, 0.0024354, 0.0000336, 0.0023050, -0.0018835, 0.0021256
2: 0.0095252, 0.0147982, 0.0098165, 0.0148910, -0.0047488, 0.0042079
3: 0.0010983, 0.0033204, 0.0010593, 0.0031977, -0.0017732, 0.0020012
4: 1.0010115, 1.0096322, 1.0008599, 1.0091560, -0.0068795, 0.0077638
5: 0.0024382, 0.0041153, 0.0024087, 0.0040226, -0.0013383, 0.0015104
6: -0.0110984, -0.0089159, -0.0109778, -0.0088775, -0.0019655, 0.0017416
7: -0.0102191, -0.0099407, -0.0102037, -0.0099358, -0.0002507, 0.0002222
8: -0.0046939, -0.0031860, -0.0047205, -0.0032693, -0.0012033, 0.0013580
9: -0.0022209, 0.0053281, -0.0018038, 0.0054609, -0.0067985, 0.0060242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0031530
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0031530
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0030850, upper bound: 0.0031689
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0031583, upper bound: 0.0031689
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0030850, upper bound: 0.0033319
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0031583, upper bound: 0.0033319
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0030840
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0030840
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0031530
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0033319, upper bound: 0.0031530

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040165, -0.0036564, -0.0040165, -0.0036444, -0.0002602, 0.0002442
1: -0.0000068, 0.0019871, -0.0000733, 0.0019874, -0.0013523, 0.0014406
2: 0.0105269, 0.0149814, 0.0105261, 0.0151298, -0.0032184, 0.0030211
3: 0.0010212, 0.0028983, 0.0009586, 0.0028986, -0.0012731, 0.0013562
4: 1.0007120, 1.0079947, 1.0004693, 1.0079958, -0.0049391, 0.0052616
5: 0.0023799, 0.0037967, 0.0023327, 0.0037969, -0.0009609, 0.0010236
6: -0.0106838, -0.0088401, -0.0106841, -0.0087786, -0.0013321, 0.0012504
7: -0.0101662, -0.0099310, -0.0101662, -0.0099232, -0.0001699, 0.0001595
8: -0.0047463, -0.0034725, -0.0047888, -0.0034723, -0.0008639, 0.0009203
9: -0.0007868, 0.0055904, -0.0007879, 0.0058029, -0.0046075, 0.0043251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031477, upper bound: 0.0031477
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031477, upper bound: 0.0032661
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040739, -0.0036738, -0.0040164, -0.0036582, -0.0003249, 0.0002537
1: 0.0000898, 0.0023049, 0.0000034, 0.0019869, -0.0014049, 0.0017989
2: 0.0098168, 0.0147655, 0.0105271, 0.0149585, -0.0040189, 0.0031386
3: 0.0011121, 0.0031975, 0.0010308, 0.0028982, -0.0013226, 0.0016936
4: 1.0010649, 1.0091556, 1.0007493, 1.0079942, -0.0051313, 0.0065704
5: 0.0024486, 0.0040225, 0.0023872, 0.0037966, -0.0009982, 0.0012782
6: -0.0109777, -0.0089294, -0.0106837, -0.0088495, -0.0016634, 0.0012991
7: -0.0102037, -0.0099424, -0.0101662, -0.0099322, -0.0002122, 0.0001657
8: -0.0046846, -0.0032694, -0.0047398, -0.0034726, -0.0008975, 0.0011493
9: -0.0018034, 0.0052813, -0.0007864, 0.0055577, -0.0057535, 0.0044933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031518, upper bound: 0.0030989
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031872, upper bound: 0.0031872
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040165, -0.0036564, -0.0040975, -0.0036558, -0.0002744, 0.0003517
1: -0.0000068, 0.0019871, -0.0000100, 0.0024358, -0.0019471, 0.0015195
2: 0.0105269, 0.0149814, 0.0095244, 0.0149885, -0.0033947, 0.0043500
3: 0.0010212, 0.0028983, 0.0010182, 0.0033208, -0.0018331, 0.0014305
4: 1.0007120, 1.0079947, 1.0007004, 1.0096335, -0.0071117, 0.0055499
5: 0.0023799, 0.0037967, 0.0023777, 0.0041155, -0.0013835, 0.0010797
6: -0.0106838, -0.0088401, -0.0110987, -0.0088371, -0.0014050, 0.0018004
7: -0.0101662, -0.0099310, -0.0102191, -0.0099306, -0.0001792, 0.0002297
8: -0.0047463, -0.0034725, -0.0047484, -0.0031858, -0.0012440, 0.0009708
9: -0.0007868, 0.0055904, -0.0022221, 0.0056005, -0.0048599, 0.0062276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030720, upper bound: 0.0032840
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030720, upper bound: 0.0033319
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040739, -0.0036738, -0.0040974, -0.0036712, -0.0003358, 0.0003612
1: 0.0000898, 0.0023049, 0.0000752, 0.0024354, -0.0019998, 0.0018595
2: 0.0098168, 0.0147655, 0.0095252, 0.0147982, -0.0041543, 0.0044677
3: 0.0011121, 0.0031975, 0.0010983, 0.0033204, -0.0018827, 0.0017506
4: 1.0010649, 1.0091556, 1.0010115, 1.0096322, -0.0073041, 0.0067918
5: 0.0024486, 0.0040225, 0.0024382, 0.0041153, -0.0014209, 0.0013213
6: -0.0109777, -0.0089294, -0.0110984, -0.0089159, -0.0017194, 0.0018491
7: -0.0102037, -0.0099424, -0.0102191, -0.0099407, -0.0002193, 0.0002359
8: -0.0046846, -0.0032694, -0.0046939, -0.0031860, -0.0012776, 0.0011880
9: -0.0018034, 0.0052813, -0.0022209, 0.0053281, -0.0059474, 0.0063960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030423, upper bound: 0.0031112
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030869, upper bound: 0.0032496
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0036558, -0.0040165, -0.0036564, -0.0003517, 0.0002744
1: -0.0000100, 0.0024358, -0.0000068, 0.0019871, -0.0015195, 0.0019471
2: 0.0095244, 0.0149885, 0.0105269, 0.0149814, -0.0043500, 0.0033947
3: 0.0010182, 0.0033208, 0.0010212, 0.0028983, -0.0014305, 0.0018331
4: 1.0007004, 1.0096335, 1.0007120, 1.0079947, -0.0055499, 0.0071117
5: 0.0023777, 0.0041155, 0.0023799, 0.0037967, -0.0010797, 0.0013835
6: -0.0110987, -0.0088371, -0.0106838, -0.0088401, -0.0018004, 0.0014050
7: -0.0102191, -0.0099306, -0.0101662, -0.0099310, -0.0002297, 0.0001792
8: -0.0047484, -0.0031858, -0.0047463, -0.0034725, -0.0009708, 0.0012440
9: -0.0022221, 0.0056005, -0.0007868, 0.0055904, -0.0062276, 0.0048599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032900, upper bound: 0.0030840
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032900, upper bound: 0.0030840
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0036558, -0.0040974, -0.0036682, -0.0002672, 0.0002834
1: -0.0000100, 0.0024358, 0.0000589, 0.0024355, -0.0015689, 0.0014796
2: 0.0095244, 0.0149885, 0.0095251, 0.0148345, -0.0033057, 0.0035052
3: 0.0010182, 0.0033208, 0.0010831, 0.0033205, -0.0014771, 0.0013930
4: 1.0007004, 1.0096335, 1.0009522, 1.0096323, -0.0057305, 0.0054044
5: 0.0023777, 0.0041155, 0.0024266, 0.0041153, -0.0011148, 0.0010514
6: -0.0110987, -0.0088371, -0.0110984, -0.0089009, -0.0013682, 0.0014508
7: -0.0102191, -0.0099306, -0.0102191, -0.0099387, -0.0001745, 0.0001851
8: -0.0047484, -0.0031858, -0.0047043, -0.0031860, -0.0010024, 0.0009453
9: -0.0022221, 0.0056005, -0.0022210, 0.0053800, -0.0047325, 0.0050181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032900, upper bound: 0.0030840
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032900, upper bound: 0.0030840
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0036712, -0.0040739, -0.0036738, -0.0003612, 0.0003358
1: 0.0000752, 0.0024354, 0.0000898, 0.0023049, -0.0018595, 0.0019998
2: 0.0095252, 0.0147982, 0.0098168, 0.0147655, -0.0044677, 0.0041543
3: 0.0010983, 0.0033204, 0.0011121, 0.0031975, -0.0017506, 0.0018827
4: 1.0010115, 1.0096322, 1.0010649, 1.0091556, -0.0067918, 0.0073041
5: 0.0024382, 0.0041153, 0.0024486, 0.0040225, -0.0013213, 0.0014209
6: -0.0110984, -0.0089159, -0.0109777, -0.0089294, -0.0018491, 0.0017194
7: -0.0102191, -0.0099407, -0.0102037, -0.0099424, -0.0002359, 0.0002193
8: -0.0046939, -0.0031860, -0.0046846, -0.0032694, -0.0011880, 0.0012776
9: -0.0022209, 0.0053281, -0.0018034, 0.0052813, -0.0063960, 0.0059474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031112, upper bound: 0.0030306
time: 0.64 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032496, upper bound: 0.0030816
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0036712, -0.0041550, -0.0036880, -0.0002747, 0.0003469
1: 0.0000752, 0.0024354, 0.0001683, 0.0027541, -0.0019207, 0.0015208
2: 0.0095252, 0.0147982, 0.0088132, 0.0145901, -0.0033977, 0.0042911
3: 0.0010983, 0.0033204, 0.0011861, 0.0036204, -0.0018083, 0.0014318
4: 1.0010115, 1.0096322, 1.0013517, 1.0107962, -0.0070154, 0.0055548
5: 0.0024382, 0.0041153, 0.0025044, 0.0043417, -0.0013648, 0.0010806
6: -0.0110984, -0.0089159, -0.0113930, -0.0090020, -0.0014063, 0.0017761
7: -0.0102191, -0.0099407, -0.0102566, -0.0099517, -0.0001794, 0.0002266
8: -0.0046939, -0.0031860, -0.0046344, -0.0029824, -0.0012271, 0.0009716
9: -0.0022209, 0.0053281, -0.0032401, 0.0050302, -0.0048642, 0.0061432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031112, upper bound: 0.0030306
time: 0.64 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032496, upper bound: 0.0030816
time: 0.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.22 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0031477, upper bound: 0.0031477
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0031477, upper bound: 0.0032661
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0031518, upper bound: 0.0030989
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0031872, upper bound: 0.0031872
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0030720, upper bound: 0.0032840
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0030720, upper bound: 0.0033319
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0030423, upper bound: 0.0031112
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0030869, upper bound: 0.0032496
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0032900, upper bound: 0.0030840
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0032900, upper bound: 0.0030840
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0032900, upper bound: 0.0030840
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0032900, upper bound: 0.0030840
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0031112, upper bound: 0.0030306
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0032496, upper bound: 0.0030816
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0031112, upper bound: 0.0030306
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 4, lower bound: -0.0032496, upper bound: 0.0030816

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040165, -0.0036564, -0.0040165, -0.0036564, -0.0002442, 0.0002442
1: -0.0000068, 0.0019871, -0.0000068, 0.0019871, -0.0013519, 0.0013519
2: 0.0105269, 0.0149814, 0.0105269, 0.0149814, -0.0030203, 0.0030203
3: 0.0010212, 0.0028983, 0.0010212, 0.0028983, -0.0012728, 0.0012728
4: 1.0007120, 1.0079947, 1.0007120, 1.0079947, -0.0049378, 0.0049378
5: 0.0023799, 0.0037967, 0.0023799, 0.0037967, -0.0009606, 0.0009606
6: -0.0106838, -0.0088401, -0.0106838, -0.0088401, -0.0012501, 0.0012501
7: -0.0101662, -0.0099310, -0.0101662, -0.0099310, -0.0001595, 0.0001595
8: -0.0047463, -0.0034725, -0.0047463, -0.0034725, -0.0008637, 0.0008637
9: -0.0007868, 0.0055904, -0.0007868, 0.0055904, -0.0043239, 0.0043239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030743, upper bound: 0.0029986
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030839, upper bound: 0.0030697
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040165, -0.0036564, -0.0040739, -0.0036738, -0.0002432, 0.0003183
1: -0.0000068, 0.0019871, 0.0000898, 0.0023049, -0.0017624, 0.0013468
2: 0.0105269, 0.0149814, 0.0098168, 0.0147655, -0.0030088, 0.0039373
3: 0.0010212, 0.0028983, 0.0011121, 0.0031975, -0.0016592, 0.0012679
4: 1.0007120, 1.0079947, 1.0010649, 1.0091556, -0.0064370, 0.0049191
5: 0.0023799, 0.0037967, 0.0024486, 0.0040225, -0.0012523, 0.0009570
6: -0.0106838, -0.0088401, -0.0109777, -0.0089294, -0.0012453, 0.0016296
7: -0.0101662, -0.0099310, -0.0102037, -0.0099424, -0.0001589, 0.0002079
8: -0.0047463, -0.0034725, -0.0046846, -0.0032694, -0.0011259, 0.0008604
9: -0.0007868, 0.0055904, -0.0018034, 0.0052813, -0.0043075, 0.0056367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030140, upper bound: 0.0031518
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030839, upper bound: 0.0031872
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040589, -0.0036699, -0.0040098, -0.0036590, -0.0003067, 0.0002379
1: 0.0000681, 0.0022218, 0.0000078, 0.0019501, -0.0013170, 0.0016985
2: 0.0100023, 0.0148139, 0.0106094, 0.0149487, -0.0037945, 0.0029423
3: 0.0010917, 0.0031193, 0.0010349, 0.0028635, -0.0012399, 0.0015990
4: 1.0009859, 1.0088521, 1.0007654, 1.0078597, -0.0048102, 0.0062036
5: 0.0024332, 0.0039635, 0.0023903, 0.0037704, -0.0009358, 0.0012069
6: -0.0109009, -0.0089094, -0.0106496, -0.0088536, -0.0015705, 0.0012178
7: -0.0101939, -0.0099398, -0.0101618, -0.0099327, -0.0002003, 0.0001553
8: -0.0046984, -0.0033225, -0.0047370, -0.0034961, -0.0008414, 0.0010851
9: -0.0015378, 0.0053506, -0.0006687, 0.0055436, -0.0054324, 0.0042122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029550, upper bound: 0.0028979
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030350, upper bound: 0.0029833
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0040686, -0.0036746, -0.0040153, -0.0036584, -0.0003065, 0.0002517
1: 0.0000940, 0.0022757, 0.0000043, 0.0019806, -0.0013939, 0.0016968
2: 0.0098820, 0.0147561, 0.0105412, 0.0149565, -0.0037909, 0.0031140
3: 0.0011161, 0.0031701, 0.0010317, 0.0028923, -0.0013123, 0.0015975
4: 1.0010804, 1.0090489, 1.0007528, 1.0079712, -0.0050910, 0.0061977
5: 0.0024516, 0.0040018, 0.0023878, 0.0037921, -0.0009904, 0.0012057
6: -0.0109507, -0.0089333, -0.0106778, -0.0088504, -0.0015690, 0.0012889
7: -0.0102002, -0.0099429, -0.0101654, -0.0099323, -0.0002001, 0.0001644
8: -0.0046819, -0.0032881, -0.0047392, -0.0034766, -0.0008905, 0.0010841
9: -0.0017101, 0.0052678, -0.0007663, 0.0055547, -0.0054272, 0.0044581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030267, upper bound: 0.0030387
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030716, upper bound: 0.0030716
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040165, -0.0036564, -0.0040974, -0.0036682, -0.0002601, 0.0003516
1: -0.0000068, 0.0019871, 0.0000588, 0.0024355, -0.0019467, 0.0014401
2: 0.0105269, 0.0149814, 0.0095251, 0.0148347, -0.0032174, 0.0043491
3: 0.0010212, 0.0028983, 0.0010830, 0.0033205, -0.0018327, 0.0013558
4: 1.0007120, 1.0079947, 1.0009518, 1.0096323, -0.0071103, 0.0052600
5: 0.0023799, 0.0037967, 0.0024266, 0.0041153, -0.0013832, 0.0010233
6: -0.0106838, -0.0088401, -0.0110984, -0.0089008, -0.0013316, 0.0018001
7: -0.0101662, -0.0099310, -0.0102191, -0.0099387, -0.0001699, 0.0002296
8: -0.0047463, -0.0034725, -0.0047044, -0.0031860, -0.0012437, 0.0009201
9: -0.0007868, 0.0055904, -0.0022210, 0.0053804, -0.0046061, 0.0062263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029272, upper bound: 0.0031621
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030160, upper bound: 0.0032073
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040165, -0.0036564, -0.0041550, -0.0036880, -0.0002477, 0.0004160
1: -0.0000068, 0.0019871, 0.0001683, 0.0027541, -0.0023034, 0.0013713
2: 0.0105269, 0.0149814, 0.0088132, 0.0145901, -0.0030637, 0.0051460
3: 0.0010212, 0.0028983, 0.0011861, 0.0036204, -0.0021685, 0.0012910
4: 1.0007120, 1.0079947, 1.0013517, 1.0107962, -0.0084131, 0.0050087
5: 0.0023799, 0.0037967, 0.0025044, 0.0043417, -0.0016367, 0.0009744
6: -0.0106838, -0.0088401, -0.0113930, -0.0090020, -0.0012680, 0.0021299
7: -0.0101662, -0.0099310, -0.0102566, -0.0099517, -0.0001617, 0.0002717
8: -0.0047463, -0.0034725, -0.0046344, -0.0029824, -0.0014716, 0.0008761
9: -0.0007868, 0.0055904, -0.0032401, 0.0050302, -0.0043860, 0.0073671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029272, upper bound: 0.0031686
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030160, upper bound: 0.0032497
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040589, -0.0036699, -0.0040918, -0.0036721, -0.0003173, 0.0003472
1: 0.0000681, 0.0022218, 0.0000803, 0.0024040, -0.0019226, 0.0017568
2: 0.0100023, 0.0148139, 0.0095954, 0.0147868, -0.0039249, 0.0042952
3: 0.0010917, 0.0031193, 0.0011031, 0.0032908, -0.0018100, 0.0016540
4: 1.0009859, 1.0088521, 1.0010301, 1.0095174, -0.0070221, 0.0064168
5: 0.0024332, 0.0039635, 0.0024418, 0.0040929, -0.0013661, 0.0012483
6: -0.0109009, -0.0089094, -0.0110693, -0.0089206, -0.0016245, 0.0017778
7: -0.0101939, -0.0099398, -0.0102154, -0.0099413, -0.0002072, 0.0002268
8: -0.0046984, -0.0033225, -0.0046907, -0.0032061, -0.0012283, 0.0011224
9: -0.0015378, 0.0053506, -0.0021204, 0.0053118, -0.0056190, 0.0061491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027161, upper bound: 0.0026782
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029203, upper bound: 0.0029979
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0040686, -0.0036746, -0.0040959, -0.0036713, -0.0003225, 0.0003587
1: 0.0000940, 0.0022757, 0.0000762, 0.0024270, -0.0019861, 0.0017855
2: 0.0098820, 0.0147561, 0.0095441, 0.0147959, -0.0039890, 0.0044371
3: 0.0011161, 0.0031701, 0.0010993, 0.0033125, -0.0018698, 0.0016810
4: 1.0010804, 1.0090489, 1.0010153, 1.0096014, -0.0072541, 0.0065216
5: 0.0024516, 0.0040018, 0.0024389, 0.0041093, -0.0014112, 0.0012687
6: -0.0109507, -0.0089333, -0.0110905, -0.0089168, -0.0016510, 0.0018365
7: -0.0102002, -0.0099429, -0.0102181, -0.0099408, -0.0002106, 0.0002343
8: -0.0046819, -0.0032881, -0.0046933, -0.0031914, -0.0012689, 0.0011407
9: -0.0017101, 0.0052678, -0.0021939, 0.0053248, -0.0057108, 0.0063522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028432, upper bound: 0.0029486
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029660, upper bound: 0.0031347
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0036682, -0.0040165, -0.0036564, -0.0003516, 0.0002601
1: 0.0000588, 0.0024355, -0.0000068, 0.0019871, -0.0014401, 0.0019467
2: 0.0095251, 0.0148347, 0.0105269, 0.0149814, -0.0043491, 0.0032174
3: 0.0010830, 0.0033205, 0.0010212, 0.0028983, -0.0013558, 0.0018327
4: 1.0009518, 1.0096323, 1.0007120, 1.0079947, -0.0052600, 0.0071103
5: 0.0024266, 0.0041153, 0.0023799, 0.0037967, -0.0010233, 0.0013832
6: -0.0110984, -0.0089008, -0.0106838, -0.0088401, -0.0018001, 0.0013316
7: -0.0102191, -0.0099387, -0.0101662, -0.0099310, -0.0002296, 0.0001699
8: -0.0047044, -0.0031860, -0.0047463, -0.0034725, -0.0009201, 0.0012437
9: -0.0022210, 0.0053804, -0.0007868, 0.0055904, -0.0062263, 0.0046061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031621, upper bound: 0.0029272
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032074, upper bound: 0.0030160
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041550, -0.0036880, -0.0040165, -0.0036564, -0.0004160, 0.0002477
1: 0.0001683, 0.0027541, -0.0000068, 0.0019871, -0.0013713, 0.0023034
2: 0.0088132, 0.0145901, 0.0105269, 0.0149814, -0.0051460, 0.0030637
3: 0.0011861, 0.0036204, 0.0010212, 0.0028983, -0.0012910, 0.0021685
4: 1.0013517, 1.0107962, 1.0007120, 1.0079947, -0.0050087, 0.0084131
5: 0.0025044, 0.0043417, 0.0023799, 0.0037967, -0.0009744, 0.0016367
6: -0.0113930, -0.0090020, -0.0106838, -0.0088401, -0.0021299, 0.0012680
7: -0.0102566, -0.0099517, -0.0101662, -0.0099310, -0.0002717, 0.0001617
8: -0.0046344, -0.0029824, -0.0047463, -0.0034725, -0.0008761, 0.0014716
9: -0.0032401, 0.0050302, -0.0007868, 0.0055904, -0.0073671, 0.0043860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031621, upper bound: 0.0029272
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032074, upper bound: 0.0030160
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0036682, -0.0040974, -0.0036682, -0.0002672, 0.0002677
1: 0.0000588, 0.0024355, 0.0000589, 0.0024355, -0.0014824, 0.0014792
2: 0.0095251, 0.0148347, 0.0095251, 0.0148345, -0.0033047, 0.0033119
3: 0.0010830, 0.0033205, 0.0010831, 0.0033205, -0.0013956, 0.0013926
4: 1.0009518, 1.0096323, 1.0009522, 1.0096323, -0.0054145, 0.0054028
5: 0.0024266, 0.0041153, 0.0024266, 0.0041153, -0.0010533, 0.0010511
6: -0.0110984, -0.0089008, -0.0110984, -0.0089009, -0.0013678, 0.0013708
7: -0.0102191, -0.0099387, -0.0102191, -0.0099387, -0.0001745, 0.0001749
8: -0.0047044, -0.0031860, -0.0047043, -0.0031860, -0.0009471, 0.0009450
9: -0.0022210, 0.0053804, -0.0022210, 0.0053800, -0.0047311, 0.0047414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031621, upper bound: 0.0029250
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032074, upper bound: 0.0030157
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041550, -0.0036880, -0.0040974, -0.0036682, -0.0003400, 0.0002646
1: 0.0001683, 0.0027541, 0.0000589, 0.0024355, -0.0014651, 0.0018824
2: 0.0088132, 0.0145901, 0.0095251, 0.0148345, -0.0042054, 0.0032731
3: 0.0011861, 0.0036204, 0.0010831, 0.0033205, -0.0013793, 0.0017722
4: 1.0013517, 1.0107962, 1.0009522, 1.0096323, -0.0053512, 0.0068754
5: 0.0025044, 0.0043417, 0.0024266, 0.0041153, -0.0010410, 0.0013375
6: -0.0113930, -0.0090020, -0.0110984, -0.0089009, -0.0017406, 0.0013547
7: -0.0102566, -0.0099517, -0.0102191, -0.0099387, -0.0002220, 0.0001728
8: -0.0046344, -0.0029824, -0.0047043, -0.0031860, -0.0009360, 0.0012026
9: -0.0032401, 0.0050302, -0.0022210, 0.0053800, -0.0060206, 0.0046859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031621, upper bound: 0.0029250
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032074, upper bound: 0.0030157
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040918, -0.0036721, -0.0040589, -0.0036699, -0.0003472, 0.0003173
1: 0.0000803, 0.0024040, 0.0000681, 0.0022218, -0.0017568, 0.0019226
2: 0.0095954, 0.0147868, 0.0100023, 0.0148139, -0.0042952, 0.0039249
3: 0.0011031, 0.0032908, 0.0010917, 0.0031193, -0.0016540, 0.0018100
4: 1.0010301, 1.0095174, 1.0009859, 1.0088521, -0.0064168, 0.0070221
5: 0.0024418, 0.0040929, 0.0024332, 0.0039635, -0.0012483, 0.0013661
6: -0.0110693, -0.0089206, -0.0109009, -0.0089094, -0.0017778, 0.0016245
7: -0.0102154, -0.0099413, -0.0101939, -0.0099398, -0.0002268, 0.0002072
8: -0.0046907, -0.0032061, -0.0046984, -0.0033225, -0.0011224, 0.0012283
9: -0.0021204, 0.0053118, -0.0015378, 0.0053506, -0.0061491, 0.0056190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026782, upper bound: 0.0027161
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029979, upper bound: 0.0029203
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040959, -0.0036713, -0.0040686, -0.0036746, -0.0003587, 0.0003225
1: 0.0000762, 0.0024270, 0.0000940, 0.0022757, -0.0017855, 0.0019861
2: 0.0095441, 0.0147959, 0.0098820, 0.0147561, -0.0044371, 0.0039890
3: 0.0010993, 0.0033125, 0.0011161, 0.0031701, -0.0016810, 0.0018698
4: 1.0010153, 1.0096014, 1.0010804, 1.0090489, -0.0065216, 0.0072541
5: 0.0024389, 0.0041093, 0.0024516, 0.0040018, -0.0012687, 0.0014112
6: -0.0110905, -0.0089168, -0.0109507, -0.0089333, -0.0018365, 0.0016510
7: -0.0102181, -0.0099408, -0.0102002, -0.0099429, -0.0002343, 0.0002106
8: -0.0046933, -0.0031914, -0.0046819, -0.0032881, -0.0011407, 0.0012689
9: -0.0021939, 0.0053248, -0.0017101, 0.0052678, -0.0063522, 0.0057108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029486, upper bound: 0.0028432
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031347, upper bound: 0.0029659
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040918, -0.0036721, -0.0041424, -0.0036774, -0.0002581, 0.0003284
1: 0.0000803, 0.0024040, 0.0001100, 0.0026843, -0.0018184, 0.0014290
2: 0.0095954, 0.0147868, 0.0089691, 0.0147205, -0.0031926, 0.0040625
3: 0.0011031, 0.0032908, 0.0011311, 0.0035547, -0.0017119, 0.0013454
4: 1.0010301, 1.0095174, 1.0011386, 1.0105413, -0.0066417, 0.0052195
5: 0.0024418, 0.0040929, 0.0024629, 0.0042921, -0.0012921, 0.0010154
6: -0.0110693, -0.0089206, -0.0113285, -0.0089480, -0.0013214, 0.0016814
7: -0.0102154, -0.0099413, -0.0102484, -0.0099448, -0.0001686, 0.0002145
8: -0.0046907, -0.0032061, -0.0046717, -0.0030270, -0.0011617, 0.0009130
9: -0.0021204, 0.0053118, -0.0030169, 0.0052169, -0.0045705, 0.0058160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026535, upper bound: 0.0026927
time: 0.67 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029979, upper bound: 0.0029130
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040959, -0.0036713, -0.0041483, -0.0036889, -0.0002725, 0.0003258
1: 0.0000762, 0.0024270, 0.0001734, 0.0027173, -0.0018037, 0.0015091
2: 0.0095441, 0.0147959, 0.0088954, 0.0145788, -0.0033714, 0.0040296
3: 0.0010993, 0.0033125, 0.0011908, 0.0035858, -0.0016981, 0.0014207
4: 1.0010153, 1.0096014, 1.0013702, 1.0106620, -0.0065880, 0.0055118
5: 0.0024389, 0.0041093, 0.0025080, 0.0043156, -0.0012816, 0.0010723
6: -0.0110905, -0.0089168, -0.0113590, -0.0090067, -0.0013954, 0.0016678
7: -0.0102181, -0.0099408, -0.0102523, -0.0099522, -0.0001780, 0.0002127
8: -0.0046933, -0.0031914, -0.0046312, -0.0030059, -0.0011523, 0.0009641
9: -0.0021939, 0.0053248, -0.0031225, 0.0050140, -0.0048266, 0.0057689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029486, upper bound: 0.0028379
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031347, upper bound: 0.0029636
time: 0.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.39 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0030743, upper bound: 0.0029986
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0030839, upper bound: 0.0030697
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0030140, upper bound: 0.0031518
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0030839, upper bound: 0.0031872
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0029550, upper bound: 0.0028979
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0030350, upper bound: 0.0029833
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0030267, upper bound: 0.0030387
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0030716, upper bound: 0.0030716
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0029272, upper bound: 0.0031621
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0030160, upper bound: 0.0032073
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0029272, upper bound: 0.0031686
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0030160, upper bound: 0.0032497
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0027161, upper bound: 0.0026782
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0029203, upper bound: 0.0029979
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0028432, upper bound: 0.0029486
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0029660, upper bound: 0.0031347
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0031621, upper bound: 0.0029272
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0032074, upper bound: 0.0030160
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0031621, upper bound: 0.0029272
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0032074, upper bound: 0.0030160
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0031621, upper bound: 0.0029250
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0032074, upper bound: 0.0030157
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0031621, upper bound: 0.0029250
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0032074, upper bound: 0.0030157
IS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0026782, upper bound: 0.0027161
IS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0029979, upper bound: 0.0029203
IS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0029486, upper bound: 0.0028432
IS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0031347, upper bound: 0.0029659
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0026535, upper bound: 0.0026927
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0029979, upper bound: 0.0029130
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0029486, upper bound: 0.0028379
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -0.0031347, upper bound: 0.0029636

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040002, -0.0036541, -0.0040098, -0.0036572, -0.0002230, 0.0002289
1: -0.0000191, 0.0018971, -0.0000022, 0.0019502, -0.0012674, 0.0012347
2: 0.0107278, 0.0150089, 0.0106091, 0.0149711, -0.0027584, 0.0028316
3: 0.0010096, 0.0028136, 0.0010255, 0.0028637, -0.0011932, 0.0011624
4: 1.0006670, 1.0076660, 1.0007288, 1.0078602, -0.0046293, 0.0045097
5: 0.0023712, 0.0037327, 0.0023832, 0.0037705, -0.0009006, 0.0008773
6: -0.0106006, -0.0088287, -0.0106497, -0.0088443, -0.0011417, 0.0011720
7: -0.0101556, -0.0099295, -0.0101618, -0.0099315, -0.0001456, 0.0001495
8: -0.0047542, -0.0035300, -0.0047434, -0.0034960, -0.0008097, 0.0007888
9: -0.0004991, 0.0056298, -0.0006691, 0.0055756, -0.0039491, 0.0040538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029733, upper bound: 0.0029556
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029966, upper bound: 0.0029429
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040118, -0.0036570, -0.0040153, -0.0036565, -0.0002180, 0.0002422
1: -0.0000030, 0.0019611, -0.0000059, 0.0019808, -0.0013413, 0.0012073
2: 0.0105848, 0.0149728, 0.0105409, 0.0149794, -0.0026973, 0.0029965
3: 0.0010248, 0.0028739, 0.0010220, 0.0028924, -0.0012628, 0.0011366
4: 1.0007261, 1.0078998, 1.0007153, 1.0079716, -0.0048990, 0.0044097
5: 0.0023826, 0.0037782, 0.0023806, 0.0037922, -0.0009531, 0.0008579
6: -0.0106598, -0.0088436, -0.0106780, -0.0088409, -0.0011164, 0.0012403
7: -0.0101631, -0.0099314, -0.0101654, -0.0099311, -0.0001424, 0.0001582
8: -0.0047439, -0.0034891, -0.0047457, -0.0034765, -0.0008569, 0.0007713
9: -0.0007038, 0.0055781, -0.0007667, 0.0055875, -0.0038615, 0.0042899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029827, upper bound: 0.0030234
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030014, upper bound: 0.0030014
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040098, -0.0036572, -0.0040589, -0.0036699, -0.0002342, 0.0003002
1: -0.0000022, 0.0019502, 0.0000681, 0.0022218, -0.0016621, 0.0012967
2: 0.0106091, 0.0149711, 0.0100023, 0.0148139, -0.0028969, 0.0037133
3: 0.0010255, 0.0028637, 0.0010917, 0.0031193, -0.0015648, 0.0012208
4: 1.0007288, 1.0078602, 1.0009859, 1.0088521, -0.0060708, 0.0047361
5: 0.0023832, 0.0037705, 0.0024332, 0.0039635, -0.0011810, 0.0009214
6: -0.0106497, -0.0088443, -0.0109009, -0.0089094, -0.0011990, 0.0015369
7: -0.0101618, -0.0099315, -0.0101939, -0.0099398, -0.0001529, 0.0001960
8: -0.0047434, -0.0034960, -0.0046984, -0.0033225, -0.0010619, 0.0008284
9: -0.0006691, 0.0055756, -0.0015378, 0.0053506, -0.0041473, 0.0053161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0029559
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028930, upper bound: 0.0030350
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040153, -0.0036565, -0.0040686, -0.0036746, -0.0002412, 0.0002963
1: -0.0000059, 0.0019808, 0.0000940, 0.0022757, -0.0016408, 0.0013356
2: 0.0105409, 0.0149794, 0.0098820, 0.0147561, -0.0029838, 0.0036657
3: 0.0010220, 0.0028924, 0.0011161, 0.0031701, -0.0015447, 0.0012574
4: 1.0007153, 1.0079716, 1.0010804, 1.0090489, -0.0059930, 0.0048782
5: 0.0023806, 0.0037922, 0.0024516, 0.0040018, -0.0011659, 0.0009490
6: -0.0106780, -0.0088409, -0.0109507, -0.0089333, -0.0012350, 0.0015172
7: -0.0101654, -0.0099311, -0.0102002, -0.0099429, -0.0001575, 0.0001935
8: -0.0047457, -0.0034765, -0.0046819, -0.0032881, -0.0010483, 0.0008533
9: -0.0007667, 0.0055875, -0.0017101, 0.0052678, -0.0042717, 0.0052479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029759, upper bound: 0.0030267
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029632, upper bound: 0.0030716
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040587, -0.0036762, -0.0040228, -0.0036731, -0.0002863, 0.0002297
1: 0.0001028, 0.0022211, 0.0000858, 0.0020223, -0.0012717, 0.0015854
2: 0.0100039, 0.0147364, 0.0104481, 0.0147745, -0.0035420, 0.0028410
3: 0.0011244, 0.0031187, 0.0011083, 0.0029315, -0.0011972, 0.0014926
4: 1.0011126, 1.0088496, 1.0010504, 1.0081233, -0.0046448, 0.0057907
5: 0.0024578, 0.0039630, 0.0024457, 0.0038217, -0.0009036, 0.0011265
6: -0.0109002, -0.0089414, -0.0107163, -0.0089257, -0.0014660, 0.0011759
7: -0.0101938, -0.0099439, -0.0101703, -0.0099419, -0.0001870, 0.0001500
8: -0.0046763, -0.0033229, -0.0046872, -0.0034500, -0.0008124, 0.0010129
9: -0.0015355, 0.0052397, -0.0008995, 0.0052942, -0.0050708, 0.0040673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029550, upper bound: 0.0028513
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029550, upper bound: 0.0028513
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040588, -0.0036718, -0.0040097, -0.0036678, -0.0002892, 0.0002365
1: 0.0000786, 0.0022217, 0.0000564, 0.0019495, -0.0013097, 0.0016013
2: 0.0100027, 0.0147906, 0.0106108, 0.0148401, -0.0035774, 0.0029259
3: 0.0011015, 0.0031192, 0.0010807, 0.0028629, -0.0012330, 0.0015075
4: 1.0010238, 1.0088516, 1.0009429, 1.0078574, -0.0047836, 0.0058486
5: 0.0024406, 0.0039634, 0.0024248, 0.0037700, -0.0009306, 0.0011378
6: -0.0109007, -0.0089190, -0.0106490, -0.0088985, -0.0014807, 0.0012110
7: -0.0101938, -0.0099411, -0.0101617, -0.0099384, -0.0001889, 0.0001545
8: -0.0046918, -0.0033226, -0.0047059, -0.0034965, -0.0008367, 0.0010230
9: -0.0015373, 0.0053173, -0.0006667, 0.0053882, -0.0051215, 0.0041888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030350, upper bound: 0.0028799
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030350, upper bound: 0.0028799
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040685, -0.0036809, -0.0040282, -0.0036724, -0.0002875, 0.0002423
1: 0.0001289, 0.0022751, 0.0000822, 0.0020519, -0.0013413, 0.0015921
2: 0.0098834, 0.0146782, 0.0103820, 0.0147825, -0.0035570, 0.0029967
3: 0.0011489, 0.0031695, 0.0011050, 0.0029594, -0.0012628, 0.0014989
4: 1.0012077, 1.0090467, 1.0010372, 1.0082314, -0.0048993, 0.0058153
5: 0.0024763, 0.0040013, 0.0024432, 0.0038427, -0.0009531, 0.0011313
6: -0.0109501, -0.0089655, -0.0107437, -0.0089224, -0.0014722, 0.0012403
7: -0.0102001, -0.0099470, -0.0101738, -0.0099415, -0.0001878, 0.0001582
8: -0.0046596, -0.0032885, -0.0046894, -0.0034311, -0.0008570, 0.0010172
9: -0.0017081, 0.0051564, -0.0009942, 0.0053056, -0.0050923, 0.0042902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030267, upper bound: 0.0029567
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030267, upper bound: 0.0029567
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040686, -0.0036765, -0.0040152, -0.0036671, -0.0002914, 0.0002505
1: 0.0001048, 0.0022756, 0.0000529, 0.0019800, -0.0013872, 0.0016136
2: 0.0098822, 0.0147319, 0.0105425, 0.0148479, -0.0036051, 0.0030992
3: 0.0011263, 0.0031700, 0.0010774, 0.0028917, -0.0013060, 0.0015192
4: 1.0011199, 1.0090485, 1.0009303, 1.0079689, -0.0050668, 0.0058938
5: 0.0024593, 0.0040017, 0.0024224, 0.0037917, -0.0009857, 0.0011466
6: -0.0109506, -0.0089433, -0.0106773, -0.0088953, -0.0014921, 0.0012827
7: -0.0102002, -0.0099442, -0.0101653, -0.0099380, -0.0001903, 0.0001636
8: -0.0046750, -0.0032881, -0.0047081, -0.0034770, -0.0008863, 0.0010309
9: -0.0017097, 0.0052332, -0.0007644, 0.0053992, -0.0051611, 0.0044369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030716, upper bound: 0.0029503
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030716, upper bound: 0.0029503
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040098, -0.0036572, -0.0040839, -0.0036599, -0.0002513, 0.0003342
1: -0.0000022, 0.0019502, 0.0000126, 0.0023606, -0.0018506, 0.0013913
2: 0.0106091, 0.0149711, 0.0096922, 0.0149379, -0.0031084, 0.0041345
3: 0.0010255, 0.0028637, 0.0010395, 0.0032500, -0.0017423, 0.0013099
4: 1.0007288, 1.0078602, 1.0007831, 1.0093592, -0.0067594, 0.0050819
5: 0.0023832, 0.0037705, 0.0023937, 0.0040621, -0.0013150, 0.0009886
6: -0.0106497, -0.0088443, -0.0110292, -0.0088580, -0.0012866, 0.0017113
7: -0.0101618, -0.0099315, -0.0102102, -0.0099333, -0.0001641, 0.0002183
8: -0.0047434, -0.0034960, -0.0047339, -0.0032338, -0.0011823, 0.0008889
9: -0.0006691, 0.0055756, -0.0019817, 0.0055282, -0.0044501, 0.0059191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0031466
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028996, upper bound: 0.0032015
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040153, -0.0036565, -0.0040912, -0.0036690, -0.0002581, 0.0003307
1: -0.0000059, 0.0019808, 0.0000632, 0.0024011, -0.0018311, 0.0014292
2: 0.0105409, 0.0149794, 0.0096017, 0.0148250, -0.0031929, 0.0040908
3: 0.0010220, 0.0028924, 0.0010871, 0.0032882, -0.0017239, 0.0013455
4: 1.0007153, 1.0079716, 1.0009677, 1.0095071, -0.0066880, 0.0052200
5: 0.0023806, 0.0037922, 0.0024297, 0.0040909, -0.0013011, 0.0010155
6: -0.0106780, -0.0088409, -0.0110667, -0.0089048, -0.0013215, 0.0016932
7: -0.0101654, -0.0099311, -0.0102150, -0.0099392, -0.0001686, 0.0002160
8: -0.0047457, -0.0034765, -0.0047016, -0.0032079, -0.0011698, 0.0009131
9: -0.0007667, 0.0055875, -0.0021113, 0.0053664, -0.0045710, 0.0058565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029657, upper bound: 0.0031826
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029608, upper bound: 0.0032136
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040098, -0.0036572, -0.0041424, -0.0036774, -0.0002427, 0.0004001
1: -0.0000022, 0.0019502, 0.0001100, 0.0026843, -0.0022151, 0.0013440
2: 0.0106091, 0.0149711, 0.0089691, 0.0147205, -0.0030026, 0.0049488
3: 0.0010255, 0.0028637, 0.0011311, 0.0035547, -0.0020854, 0.0012653
4: 1.0007288, 1.0078602, 1.0011386, 1.0105413, -0.0080907, 0.0049089
5: 0.0023832, 0.0037705, 0.0024629, 0.0042921, -0.0015740, 0.0009550
6: -0.0106497, -0.0088443, -0.0113285, -0.0089480, -0.0012428, 0.0020483
7: -0.0101618, -0.0099315, -0.0102484, -0.0099448, -0.0001585, 0.0002613
8: -0.0047434, -0.0034960, -0.0046717, -0.0030270, -0.0014152, 0.0008587
9: -0.0006691, 0.0055756, -0.0030169, 0.0052169, -0.0042986, 0.0070848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0028134
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028048, upper bound: 0.0030564
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040153, -0.0036565, -0.0041483, -0.0036889, -0.0002455, 0.0003970
1: -0.0000059, 0.0019808, 0.0001734, 0.0027173, -0.0021980, 0.0013595
2: 0.0105409, 0.0149794, 0.0088954, 0.0145788, -0.0030373, 0.0049106
3: 0.0010220, 0.0028924, 0.0011908, 0.0035858, -0.0020694, 0.0012799
4: 1.0007153, 1.0079716, 1.0013702, 1.0106620, -0.0080283, 0.0049656
5: 0.0023806, 0.0037922, 0.0025080, 0.0043156, -0.0015618, 0.0009660
6: -0.0106780, -0.0088409, -0.0113590, -0.0090067, -0.0012571, 0.0020325
7: -0.0101654, -0.0099311, -0.0102523, -0.0099522, -0.0001604, 0.0002593
8: -0.0047457, -0.0034765, -0.0046312, -0.0030059, -0.0014043, 0.0008686
9: -0.0007667, 0.0055875, -0.0031225, 0.0050140, -0.0043483, 0.0070302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028417, upper bound: 0.0029715
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028955, upper bound: 0.0031347
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040587, -0.0036762, -0.0041050, -0.0036871, -0.0002972, 0.0003424
1: 0.0001028, 0.0022211, 0.0001633, 0.0024772, -0.0018960, 0.0016457
2: 0.0100039, 0.0147364, 0.0094319, 0.0146013, -0.0036767, 0.0042359
3: 0.0011244, 0.0031187, 0.0011813, 0.0033597, -0.0017850, 0.0015494
4: 1.0011126, 1.0088496, 1.0013334, 1.0097848, -0.0069252, 0.0060110
5: 0.0024578, 0.0039630, 0.0025008, 0.0041449, -0.0013472, 0.0011694
6: -0.0109002, -0.0089414, -0.0111370, -0.0089974, -0.0015218, 0.0017532
7: -0.0101938, -0.0099439, -0.0102240, -0.0099511, -0.0001941, 0.0002236
8: -0.0046763, -0.0033229, -0.0046376, -0.0031593, -0.0012113, 0.0010514
9: -0.0015355, 0.0052397, -0.0023545, 0.0050462, -0.0052637, 0.0060642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027161, upper bound: 0.0026782
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027161, upper bound: 0.0026782
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040588, -0.0036718, -0.0040916, -0.0036801, -0.0003005, 0.0003459
1: 0.0000786, 0.0022217, 0.0001244, 0.0024032, -0.0019152, 0.0016639
2: 0.0100027, 0.0147906, 0.0095970, 0.0146881, -0.0037173, 0.0042787
3: 0.0011015, 0.0031192, 0.0011447, 0.0032901, -0.0018030, 0.0015665
4: 1.0010238, 1.0088516, 1.0011915, 1.0095148, -0.0069951, 0.0060773
5: 0.0024406, 0.0039634, 0.0024732, 0.0040924, -0.0013608, 0.0011823
6: -0.0109007, -0.0089190, -0.0110686, -0.0089614, -0.0015386, 0.0017709
7: -0.0101938, -0.0099411, -0.0102153, -0.0099465, -0.0001963, 0.0002259
8: -0.0046918, -0.0033226, -0.0046625, -0.0032066, -0.0012236, 0.0010630
9: -0.0015373, 0.0053173, -0.0021180, 0.0051705, -0.0053217, 0.0061254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029203, upper bound: 0.0029787
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029203, upper bound: 0.0029787
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040685, -0.0036809, -0.0041090, -0.0036863, -0.0003033, 0.0003523
1: 0.0001289, 0.0022751, 0.0001590, 0.0024996, -0.0019507, 0.0016792
2: 0.0098834, 0.0146782, 0.0093818, 0.0146110, -0.0037516, 0.0043580
3: 0.0011489, 0.0031695, 0.0011772, 0.0033808, -0.0018365, 0.0015809
4: 1.0012077, 1.0090467, 1.0013175, 1.0098667, -0.0071248, 0.0061334
5: 0.0024763, 0.0040013, 0.0024977, 0.0041609, -0.0013861, 0.0011932
6: -0.0109501, -0.0089655, -0.0111577, -0.0089934, -0.0015528, 0.0018038
7: -0.0102001, -0.0099470, -0.0102266, -0.0099505, -0.0001981, 0.0002301
8: -0.0046596, -0.0032885, -0.0046404, -0.0031450, -0.0012462, 0.0010728
9: -0.0017081, 0.0051564, -0.0024262, 0.0050602, -0.0053708, 0.0062390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028432, upper bound: 0.0029471
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028432, upper bound: 0.0029471
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040686, -0.0036765, -0.0040958, -0.0036793, -0.0003076, 0.0003575
1: 0.0001048, 0.0022756, 0.0001205, 0.0024263, -0.0019793, 0.0017031
2: 0.0098822, 0.0147319, 0.0095456, 0.0146970, -0.0038049, 0.0044220
3: 0.0011263, 0.0031700, 0.0011410, 0.0033118, -0.0018635, 0.0016034
4: 1.0011199, 1.0090485, 1.0011770, 1.0095989, -0.0072295, 0.0062206
5: 0.0024593, 0.0040017, 0.0024704, 0.0041088, -0.0014064, 0.0012102
6: -0.0109506, -0.0089433, -0.0110899, -0.0089578, -0.0015748, 0.0018303
7: -0.0102002, -0.0099442, -0.0102180, -0.0099460, -0.0002009, 0.0002335
8: -0.0046750, -0.0032881, -0.0046650, -0.0031919, -0.0012646, 0.0010881
9: -0.0017097, 0.0052332, -0.0021916, 0.0051832, -0.0054472, 0.0063307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029660, upper bound: 0.0030876
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029660, upper bound: 0.0030876
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0036599, -0.0040098, -0.0036572, -0.0003342, 0.0002513
1: 0.0000126, 0.0023606, -0.0000022, 0.0019502, -0.0013913, 0.0018506
2: 0.0096922, 0.0149379, 0.0106091, 0.0149711, -0.0041345, 0.0031084
3: 0.0010395, 0.0032500, 0.0010255, 0.0028637, -0.0013099, 0.0017423
4: 1.0007831, 1.0093592, 1.0007288, 1.0078602, -0.0050819, 0.0067594
5: 0.0023937, 0.0040621, 0.0023832, 0.0037705, -0.0009886, 0.0013150
6: -0.0110292, -0.0088580, -0.0106497, -0.0088443, -0.0017113, 0.0012866
7: -0.0102102, -0.0099333, -0.0101618, -0.0099315, -0.0002183, 0.0001641
8: -0.0047339, -0.0032338, -0.0047434, -0.0034960, -0.0008889, 0.0011823
9: -0.0019817, 0.0055282, -0.0006691, 0.0055756, -0.0059191, 0.0044501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031466, upper bound: 0.0028824
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032015, upper bound: 0.0028996
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040912, -0.0036690, -0.0040153, -0.0036565, -0.0003307, 0.0002581
1: 0.0000632, 0.0024011, -0.0000059, 0.0019808, -0.0014292, 0.0018311
2: 0.0096017, 0.0148250, 0.0105409, 0.0149794, -0.0040908, 0.0031929
3: 0.0010871, 0.0032882, 0.0010220, 0.0028924, -0.0013455, 0.0017239
4: 1.0009677, 1.0095071, 1.0007153, 1.0079716, -0.0052200, 0.0066880
5: 0.0024297, 0.0040909, 0.0023806, 0.0037922, -0.0010155, 0.0013011
6: -0.0110667, -0.0089048, -0.0106780, -0.0088409, -0.0016932, 0.0013215
7: -0.0102150, -0.0099392, -0.0101654, -0.0099311, -0.0002160, 0.0001686
8: -0.0047016, -0.0032079, -0.0047457, -0.0034765, -0.0009131, 0.0011698
9: -0.0021113, 0.0053664, -0.0007667, 0.0055875, -0.0058565, 0.0045710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031826, upper bound: 0.0029657
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032136, upper bound: 0.0029608
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0041424, -0.0036774, -0.0040098, -0.0036572, -0.0004001, 0.0002427
1: 0.0001100, 0.0026843, -0.0000022, 0.0019502, -0.0013440, 0.0022151
2: 0.0089691, 0.0147205, 0.0106091, 0.0149711, -0.0049488, 0.0030026
3: 0.0011311, 0.0035547, 0.0010255, 0.0028637, -0.0012653, 0.0020854
4: 1.0011386, 1.0105413, 1.0007288, 1.0078602, -0.0049089, 0.0080907
5: 0.0024629, 0.0042921, 0.0023832, 0.0037705, -0.0009550, 0.0015740
6: -0.0113285, -0.0089480, -0.0106497, -0.0088443, -0.0020483, 0.0012428
7: -0.0102484, -0.0099448, -0.0101618, -0.0099315, -0.0002613, 0.0001585
8: -0.0046717, -0.0030270, -0.0047434, -0.0034960, -0.0008587, 0.0014152
9: -0.0030169, 0.0052169, -0.0006691, 0.0055756, -0.0070848, 0.0042986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028134, upper bound: 0.0027041
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030565, upper bound: 0.0028048
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041483, -0.0036889, -0.0040153, -0.0036565, -0.0003970, 0.0002455
1: 0.0001734, 0.0027173, -0.0000059, 0.0019808, -0.0013595, 0.0021980
2: 0.0088954, 0.0145788, 0.0105409, 0.0149794, -0.0049106, 0.0030373
3: 0.0011908, 0.0035858, 0.0010220, 0.0028924, -0.0012799, 0.0020694
4: 1.0013702, 1.0106620, 1.0007153, 1.0079716, -0.0049656, 0.0080283
5: 0.0025080, 0.0043156, 0.0023806, 0.0037922, -0.0009660, 0.0015618
6: -0.0113590, -0.0090067, -0.0106780, -0.0088409, -0.0020325, 0.0012571
7: -0.0102523, -0.0099522, -0.0101654, -0.0099311, -0.0002593, 0.0001604
8: -0.0046312, -0.0030059, -0.0047457, -0.0034765, -0.0008686, 0.0014043
9: -0.0031225, 0.0050140, -0.0007667, 0.0055875, -0.0070302, 0.0043483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029715, upper bound: 0.0028417
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031347, upper bound: 0.0028955
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0036599, -0.0040918, -0.0036692, -0.0002462, 0.0002513
1: 0.0000126, 0.0023606, 0.0000640, 0.0024040, -0.0013912, 0.0013630
2: 0.0096922, 0.0149379, 0.0095953, 0.0148231, -0.0030451, 0.0031082
3: 0.0010395, 0.0032500, 0.0010879, 0.0032909, -0.0013098, 0.0012832
4: 1.0007831, 1.0093592, 1.0009708, 1.0095177, -0.0050815, 0.0049784
5: 0.0023937, 0.0040621, 0.0024303, 0.0040930, -0.0009886, 0.0009685
6: -0.0110292, -0.0088580, -0.0110693, -0.0089056, -0.0012603, 0.0012865
7: -0.0102102, -0.0099333, -0.0102154, -0.0099394, -0.0001608, 0.0001641
8: -0.0047339, -0.0032338, -0.0047010, -0.0032061, -0.0008888, 0.0008708
9: -0.0019817, 0.0055282, -0.0021205, 0.0053637, -0.0043594, 0.0044498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031466, upper bound: 0.0028782
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032015, upper bound: 0.0028996
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040912, -0.0036690, -0.0040959, -0.0036684, -0.0002397, 0.0002656
1: 0.0000632, 0.0024011, 0.0000600, 0.0024270, -0.0014706, 0.0013271
2: 0.0096017, 0.0148250, 0.0095440, 0.0148321, -0.0029648, 0.0032856
3: 0.0010871, 0.0032882, 0.0010841, 0.0033125, -0.0013845, 0.0012494
4: 1.0009677, 1.0095071, 1.0009561, 1.0096016, -0.0053715, 0.0048472
5: 0.0024297, 0.0040909, 0.0024274, 0.0041093, -0.0010450, 0.0009430
6: -0.0110667, -0.0089048, -0.0110906, -0.0089018, -0.0012271, 0.0013599
7: -0.0102150, -0.0099392, -0.0102181, -0.0099389, -0.0001565, 0.0001735
8: -0.0047016, -0.0032079, -0.0047036, -0.0031914, -0.0009396, 0.0008478
9: -0.0021113, 0.0053664, -0.0021940, 0.0053767, -0.0042445, 0.0047037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031826, upper bound: 0.0029644
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032136, upper bound: 0.0029608
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0041424, -0.0036774, -0.0040918, -0.0036692, -0.0003218, 0.0002550
1: 0.0001100, 0.0026843, 0.0000640, 0.0024040, -0.0014117, 0.0017816
2: 0.0089691, 0.0147205, 0.0095953, 0.0148231, -0.0039803, 0.0031539
3: 0.0011311, 0.0035547, 0.0010879, 0.0032909, -0.0013291, 0.0016773
4: 1.0011386, 1.0105413, 1.0009708, 1.0095177, -0.0051562, 0.0065073
5: 0.0024629, 0.0042921, 0.0024303, 0.0040930, -0.0010031, 0.0012659
6: -0.0113285, -0.0089480, -0.0110693, -0.0089056, -0.0016474, 0.0013054
7: -0.0102484, -0.0099448, -0.0102154, -0.0099394, -0.0002101, 0.0001665
8: -0.0046717, -0.0030270, -0.0047010, -0.0032061, -0.0009019, 0.0011382
9: -0.0030169, 0.0052169, -0.0021205, 0.0053637, -0.0056983, 0.0045152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028052, upper bound: 0.0026859
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030565, upper bound: 0.0028031
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041483, -0.0036889, -0.0040959, -0.0036684, -0.0003163, 0.0002624
1: 0.0001734, 0.0027173, 0.0000600, 0.0024270, -0.0014530, 0.0017513
2: 0.0088954, 0.0145788, 0.0095440, 0.0148321, -0.0039126, 0.0032462
3: 0.0011908, 0.0035858, 0.0010841, 0.0033125, -0.0013679, 0.0016488
4: 1.0013702, 1.0106620, 1.0009561, 1.0096016, -0.0053071, 0.0063966
5: 0.0025080, 0.0043156, 0.0024274, 0.0041093, -0.0010324, 0.0012444
6: -0.0113590, -0.0090067, -0.0110906, -0.0089018, -0.0016194, 0.0013436
7: -0.0102523, -0.0099522, -0.0102181, -0.0099389, -0.0002066, 0.0001714
8: -0.0046312, -0.0030059, -0.0047036, -0.0031914, -0.0009283, 0.0011189
9: -0.0031225, 0.0050140, -0.0021940, 0.0053767, -0.0056014, 0.0046473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029715, upper bound: 0.0028346
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031347, upper bound: 0.0028955
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0036871, -0.0040587, -0.0036762, -0.0003424, 0.0002972
1: 0.0001633, 0.0024772, 0.0001028, 0.0022211, -0.0016457, 0.0018960
2: 0.0094319, 0.0146013, 0.0100039, 0.0147364, -0.0042359, 0.0036767
3: 0.0011813, 0.0033597, 0.0011244, 0.0031187, -0.0015494, 0.0017850
4: 1.0013334, 1.0097848, 1.0011126, 1.0088496, -0.0060110, 0.0069252
5: 0.0025008, 0.0041449, 0.0024578, 0.0039630, -0.0011694, 0.0013472
6: -0.0111370, -0.0089974, -0.0109002, -0.0089414, -0.0017532, 0.0015218
7: -0.0102240, -0.0099511, -0.0101938, -0.0099439, -0.0002236, 0.0001941
8: -0.0046376, -0.0031593, -0.0046763, -0.0033229, -0.0010514, 0.0012113
9: -0.0023545, 0.0050462, -0.0015355, 0.0052397, -0.0060642, 0.0052637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026782, upper bound: 0.0027161
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026782, upper bound: 0.0027080
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040916, -0.0036801, -0.0040588, -0.0036718, -0.0003459, 0.0003005
1: 0.0001244, 0.0024032, 0.0000786, 0.0022217, -0.0016639, 0.0019152
2: 0.0095970, 0.0146881, 0.0100027, 0.0147906, -0.0042787, 0.0037173
3: 0.0011447, 0.0032901, 0.0011015, 0.0031192, -0.0015665, 0.0018030
4: 1.0011915, 1.0095148, 1.0010238, 1.0088516, -0.0060773, 0.0069951
5: 0.0024732, 0.0040924, 0.0024406, 0.0039634, -0.0011823, 0.0013608
6: -0.0110686, -0.0089614, -0.0109007, -0.0089190, -0.0017709, 0.0015386
7: -0.0102153, -0.0099465, -0.0101938, -0.0099411, -0.0002259, 0.0001963
8: -0.0046625, -0.0032066, -0.0046918, -0.0033226, -0.0010630, 0.0012236
9: -0.0021180, 0.0051705, -0.0015373, 0.0053173, -0.0061254, 0.0053217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0029203
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028601
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041090, -0.0036863, -0.0040685, -0.0036809, -0.0003523, 0.0003033
1: 0.0001590, 0.0024996, 0.0001289, 0.0022751, -0.0016792, 0.0019507
2: 0.0093818, 0.0146110, 0.0098834, 0.0146782, -0.0043580, 0.0037516
3: 0.0011772, 0.0033808, 0.0011489, 0.0031695, -0.0015809, 0.0018365
4: 1.0013175, 1.0098667, 1.0012077, 1.0090467, -0.0061334, 0.0071248
5: 0.0024977, 0.0041609, 0.0024763, 0.0040013, -0.0011932, 0.0013861
6: -0.0111577, -0.0089934, -0.0109501, -0.0089655, -0.0018038, 0.0015528
7: -0.0102266, -0.0099505, -0.0102001, -0.0099470, -0.0002301, 0.0001981
8: -0.0046404, -0.0031450, -0.0046596, -0.0032885, -0.0010728, 0.0012462
9: -0.0024262, 0.0050602, -0.0017081, 0.0051564, -0.0062390, 0.0053708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029471, upper bound: 0.0028432
time: 0.64 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029471, upper bound: 0.0027877
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040958, -0.0036793, -0.0040686, -0.0036765, -0.0003575, 0.0003076
1: 0.0001205, 0.0024263, 0.0001048, 0.0022756, -0.0017031, 0.0019793
2: 0.0095456, 0.0146970, 0.0098822, 0.0147319, -0.0044220, 0.0038049
3: 0.0011410, 0.0033118, 0.0011263, 0.0031700, -0.0016034, 0.0018635
4: 1.0011770, 1.0095989, 1.0011199, 1.0090485, -0.0062206, 0.0072295
5: 0.0024704, 0.0041088, 0.0024593, 0.0040017, -0.0012102, 0.0014064
6: -0.0110899, -0.0089578, -0.0109506, -0.0089433, -0.0018303, 0.0015748
7: -0.0102180, -0.0099460, -0.0102002, -0.0099442, -0.0002335, 0.0002009
8: -0.0046650, -0.0031919, -0.0046750, -0.0032881, -0.0010881, 0.0012646
9: -0.0021916, 0.0051832, -0.0017097, 0.0052332, -0.0063307, 0.0054472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030876, upper bound: 0.0029659
time: 0.73 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030876, upper bound: 0.0028811
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0036871, -0.0041423, -0.0036841, -0.0002494, 0.0003073
1: 0.0001633, 0.0024772, 0.0001469, 0.0026836, -0.0017018, 0.0013811
2: 0.0094319, 0.0146013, 0.0089707, 0.0146381, -0.0030856, 0.0038019
3: 0.0011813, 0.0033597, 0.0011658, 0.0035541, -0.0016021, 0.0013003
4: 1.0013334, 1.0097848, 1.0012733, 1.0105388, -0.0062156, 0.0050445
5: 0.0025008, 0.0041449, 0.0024891, 0.0042916, -0.0012092, 0.0009814
6: -0.0111370, -0.0089974, -0.0113278, -0.0089822, -0.0012771, 0.0015736
7: -0.0102240, -0.0099511, -0.0102483, -0.0099491, -0.0001629, 0.0002007
8: -0.0046376, -0.0031593, -0.0046481, -0.0030275, -0.0010872, 0.0008824
9: -0.0023545, 0.0050462, -0.0030147, 0.0050988, -0.0044174, 0.0054429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026535, upper bound: 0.0026927
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026535, upper bound: 0.0026921
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040916, -0.0036801, -0.0041424, -0.0036793, -0.0002568, 0.0003107
1: 0.0001244, 0.0024032, 0.0001204, 0.0026841, -0.0017203, 0.0014222
2: 0.0095970, 0.0146881, 0.0089695, 0.0146972, -0.0031773, 0.0038434
3: 0.0011447, 0.0032901, 0.0011409, 0.0035546, -0.0016196, 0.0013389
4: 1.0011915, 1.0095148, 1.0011766, 1.0105407, -0.0062835, 0.0051945
5: 0.0024732, 0.0040924, 0.0024703, 0.0042920, -0.0012224, 0.0010105
6: -0.0110686, -0.0089614, -0.0113284, -0.0089577, -0.0013151, 0.0015908
7: -0.0102153, -0.0099465, -0.0102484, -0.0099460, -0.0001677, 0.0002029
8: -0.0046625, -0.0032066, -0.0046651, -0.0030271, -0.0010991, 0.0009086
9: -0.0021180, 0.0051705, -0.0030164, 0.0051835, -0.0045487, 0.0055023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0029130
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028594
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041090, -0.0036863, -0.0041482, -0.0036958, -0.0002628, 0.0003065
1: 0.0001590, 0.0024996, 0.0002116, 0.0027166, -0.0016970, 0.0014551
2: 0.0093818, 0.0146110, 0.0088969, 0.0144933, -0.0032509, 0.0037914
3: 0.0011772, 0.0033808, 0.0012268, 0.0035852, -0.0015977, 0.0013699
4: 1.0013175, 1.0098667, 1.0015100, 1.0106593, -0.0061984, 0.0053149
5: 0.0024977, 0.0041609, 0.0025351, 0.0043151, -0.0012058, 0.0010340
6: -0.0111577, -0.0089934, -0.0113584, -0.0090421, -0.0013455, 0.0015692
7: -0.0102266, -0.0099505, -0.0102522, -0.0099568, -0.0001716, 0.0002002
8: -0.0046404, -0.0031450, -0.0046068, -0.0030064, -0.0010842, 0.0009297
9: -0.0024262, 0.0050602, -0.0031203, 0.0048916, -0.0046541, 0.0054278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029471, upper bound: 0.0028379
time: 0.65 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029471, upper bound: 0.0027854
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040958, -0.0036793, -0.0041483, -0.0036907, -0.0002715, 0.0003105
1: 0.0001205, 0.0024263, 0.0001833, 0.0027172, -0.0017194, 0.0015031
2: 0.0095456, 0.0146970, 0.0088957, 0.0145566, -0.0033582, 0.0038412
3: 0.0011410, 0.0033118, 0.0012002, 0.0035857, -0.0016187, 0.0014151
4: 1.0011770, 1.0095989, 1.0014064, 1.0106614, -0.0062800, 0.0054902
5: 0.0024704, 0.0041088, 0.0025150, 0.0043155, -0.0012217, 0.0010681
6: -0.0110899, -0.0089578, -0.0113589, -0.0090159, -0.0013899, 0.0015899
7: -0.0102180, -0.0099460, -0.0102523, -0.0099534, -0.0001773, 0.0002028
8: -0.0046650, -0.0031919, -0.0046249, -0.0030060, -0.0010985, 0.0009603
9: -0.0021916, 0.0051832, -0.0031220, 0.0049822, -0.0048076, 0.0054992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030876, upper bound: 0.0029636
time: 0.82 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030876, upper bound: 0.0028807
time: 0.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.65 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029733, upper bound: 0.0029556
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029966, upper bound: 0.0029429
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029827, upper bound: 0.0030234
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030014, upper bound: 0.0030014
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0029559
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028930, upper bound: 0.0030350
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029759, upper bound: 0.0030267
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029632, upper bound: 0.0030716
IS_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029550, upper bound: 0.0028513
IS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029550, upper bound: 0.0028513
IS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030350, upper bound: 0.0028799
IS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030350, upper bound: 0.0028799
IS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030267, upper bound: 0.0029567
IS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030267, upper bound: 0.0029567
IS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030716, upper bound: 0.0029503
IS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030716, upper bound: 0.0029503
IS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0031466
IS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028996, upper bound: 0.0032015
IS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029657, upper bound: 0.0031826
IS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029608, upper bound: 0.0032136
IS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0028134
IS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028048, upper bound: 0.0030564
IS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028417, upper bound: 0.0029715
IS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028955, upper bound: 0.0031347
IS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0027161, upper bound: 0.0026782
IS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0027161, upper bound: 0.0026782
IS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029203, upper bound: 0.0029787
IS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029203, upper bound: 0.0029787
IS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028432, upper bound: 0.0029471
IS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028432, upper bound: 0.0029471
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029660, upper bound: 0.0030876
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029660, upper bound: 0.0030876
IS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0031466, upper bound: 0.0028824
IS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0032015, upper bound: 0.0028996
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0031826, upper bound: 0.0029657
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0032136, upper bound: 0.0029608
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028134, upper bound: 0.0027041
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030565, upper bound: 0.0028048
IS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029715, upper bound: 0.0028417
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0031347, upper bound: 0.0028955
IS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0031466, upper bound: 0.0028782
IS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0032015, upper bound: 0.0028996
IS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0031826, upper bound: 0.0029644
IS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0032136, upper bound: 0.0029608
IS_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0028052, upper bound: 0.0026859
IS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030565, upper bound: 0.0028031
IS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029715, upper bound: 0.0028346
IS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0031347, upper bound: 0.0028955
IS_A2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0026782, upper bound: 0.0027161
IS_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0026782, upper bound: 0.0027080
IS_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0029203
IS_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028601
IS_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029471, upper bound: 0.0028432
IS_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029471, upper bound: 0.0027877
IS_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030876, upper bound: 0.0029659
IS_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030876, upper bound: 0.0028811
IS_A2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0026535, upper bound: 0.0026927
IS_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0026535, upper bound: 0.0026921
IS_A2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0029130
IS_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028594
IS_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029471, upper bound: 0.0028379
IS_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0029471, upper bound: 0.0027854
IS_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030876, upper bound: 0.0029636
IS_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 4, lower bound: -0.0030876, upper bound: 0.0028807

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040001, -0.0036604, -0.0040228, -0.0036705, -0.0002028, 0.0002208
1: 0.0000155, 0.0018964, 0.0000714, 0.0020224, -0.0012225, 0.0011228
2: 0.0107294, 0.0149315, 0.0104479, 0.0148066, -0.0025085, 0.0027313
3: 0.0010422, 0.0028130, 0.0010948, 0.0029316, -0.0011510, 0.0010571
4: 1.0007935, 1.0076636, 1.0009978, 1.0081236, -0.0044653, 0.0041011
5: 0.0023958, 0.0037323, 0.0024355, 0.0038218, -0.0008687, 0.0007978
6: -0.0105999, -0.0088607, -0.0107165, -0.0089124, -0.0010383, 0.0011305
7: -0.0101555, -0.0099336, -0.0101703, -0.0099402, -0.0001324, 0.0001442
8: -0.0047321, -0.0035304, -0.0046963, -0.0034499, -0.0007811, 0.0007174
9: -0.0004969, 0.0055190, -0.0008999, 0.0053401, -0.0035913, 0.0039102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022989, upper bound: 0.0019957
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029556
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029557
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040002, -0.0036561, -0.0040097, -0.0036659, -0.0002019, 0.0002276
1: -0.0000083, 0.0018969, 0.0000459, 0.0019496, -0.0012600, 0.0011180
2: 0.0107282, 0.0149847, 0.0106105, 0.0148637, -0.0024977, 0.0028150
3: 0.0010198, 0.0028135, 0.0010708, 0.0028631, -0.0011862, 0.0010525
4: 1.0007067, 1.0076655, 1.0009044, 1.0078579, -0.0046022, 0.0040834
5: 0.0023789, 0.0037326, 0.0024173, 0.0037701, -0.0008953, 0.0007944
6: -0.0106004, -0.0088387, -0.0106491, -0.0088888, -0.0010338, 0.0011651
7: -0.0101555, -0.0099308, -0.0101618, -0.0099372, -0.0001319, 0.0001486
8: -0.0047473, -0.0035300, -0.0047127, -0.0034964, -0.0008050, 0.0007143
9: -0.0004986, 0.0055951, -0.0006671, 0.0054219, -0.0035758, 0.0040300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026125, upper bound: 0.0023198
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029836, upper bound: 0.0029164
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029836, upper bound: 0.0029429
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040117, -0.0036630, -0.0040282, -0.0036698, -0.0001989, 0.0002329
1: 0.0000302, 0.0019605, 0.0000678, 0.0020520, -0.0012894, 0.0011011
2: 0.0105862, 0.0148986, 0.0103817, 0.0148147, -0.0024600, 0.0028806
3: 0.0010560, 0.0028733, 0.0010914, 0.0029595, -0.0012139, 0.0010366
4: 1.0008473, 1.0078976, 1.0009845, 1.0082319, -0.0047094, 0.0040217
5: 0.0024062, 0.0037778, 0.0024329, 0.0038428, -0.0009162, 0.0007824
6: -0.0106592, -0.0088743, -0.0107438, -0.0089091, -0.0010182, 0.0011923
7: -0.0101630, -0.0099354, -0.0101738, -0.0099398, -0.0001299, 0.0001521
8: -0.0047227, -0.0034894, -0.0046987, -0.0034310, -0.0008238, 0.0007035
9: -0.0007019, 0.0054718, -0.0009946, 0.0053517, -0.0035217, 0.0041239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0030140
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0030235
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040117, -0.0036590, -0.0040152, -0.0036652, -0.0001994, 0.0002411
1: 0.0000076, 0.0019610, 0.0000422, 0.0019801, -0.0013349, 0.0011038
2: 0.0105851, 0.0149491, 0.0105423, 0.0148718, -0.0024660, 0.0029823
3: 0.0010347, 0.0028738, 0.0010674, 0.0028918, -0.0012567, 0.0010392
4: 1.0007647, 1.0078994, 1.0008912, 1.0079695, -0.0048757, 0.0040316
5: 0.0023902, 0.0037781, 0.0024148, 0.0037918, -0.0009485, 0.0007843
6: -0.0106597, -0.0088534, -0.0106774, -0.0088854, -0.0010207, 0.0012344
7: -0.0101631, -0.0099327, -0.0101654, -0.0099368, -0.0001302, 0.0001575
8: -0.0047371, -0.0034891, -0.0047150, -0.0034769, -0.0008528, 0.0007052
9: -0.0007034, 0.0055442, -0.0007648, 0.0054334, -0.0035304, 0.0042695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029429, upper bound: 0.0029966
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029429, upper bound: 0.0030014
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040228, -0.0036705, -0.0040587, -0.0036762, -0.0002269, 0.0002800
1: 0.0000714, 0.0020224, 0.0001028, 0.0022211, -0.0015502, 0.0012565
2: 0.0104479, 0.0148066, 0.0100039, 0.0147364, -0.0028072, 0.0034632
3: 0.0010948, 0.0029316, 0.0011244, 0.0031187, -0.0014594, 0.0011830
4: 1.0009978, 1.0081236, 1.0011126, 1.0088496, -0.0056620, 0.0045895
5: 0.0024355, 0.0038218, 0.0024578, 0.0039630, -0.0011015, 0.0008928
6: -0.0107165, -0.0089124, -0.0109002, -0.0089414, -0.0011619, 0.0014334
7: -0.0101703, -0.0099402, -0.0101938, -0.0099439, -0.0001482, 0.0001828
8: -0.0046963, -0.0034499, -0.0046763, -0.0033229, -0.0009904, 0.0008028
9: -0.0008999, 0.0053401, -0.0015355, 0.0052397, -0.0040189, 0.0049580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0028843
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0029559
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040097, -0.0036659, -0.0040588, -0.0036718, -0.0002326, 0.0002827
1: 0.0000459, 0.0019496, 0.0000786, 0.0022217, -0.0015654, 0.0012878
2: 0.0106105, 0.0148637, 0.0100027, 0.0147906, -0.0028770, 0.0034972
3: 0.0010708, 0.0028631, 0.0011015, 0.0031192, -0.0014737, 0.0012124
4: 1.0009044, 1.0078579, 1.0010238, 1.0088516, -0.0057176, 0.0047036
5: 0.0024173, 0.0037701, 0.0024406, 0.0039634, -0.0011123, 0.0009150
6: -0.0106491, -0.0088888, -0.0109007, -0.0089190, -0.0011908, 0.0014475
7: -0.0101618, -0.0099372, -0.0101938, -0.0099411, -0.0001519, 0.0001846
8: -0.0047127, -0.0034964, -0.0046918, -0.0033226, -0.0010001, 0.0008227
9: -0.0006671, 0.0054219, -0.0015373, 0.0053173, -0.0041188, 0.0050067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028418, upper bound: 0.0029695
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028418, upper bound: 0.0030350
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040282, -0.0036698, -0.0040685, -0.0036809, -0.0002327, 0.0002771
1: 0.0000678, 0.0020520, 0.0001289, 0.0022751, -0.0015344, 0.0012886
2: 0.0103817, 0.0148147, 0.0098834, 0.0146782, -0.0028788, 0.0034281
3: 0.0010914, 0.0029595, 0.0011489, 0.0031695, -0.0014446, 0.0012131
4: 1.0009845, 1.0082319, 1.0012077, 1.0090467, -0.0056046, 0.0047065
5: 0.0024329, 0.0038428, 0.0024763, 0.0040013, -0.0010903, 0.0009156
6: -0.0107438, -0.0089091, -0.0109501, -0.0089655, -0.0011915, 0.0014189
7: -0.0101738, -0.0099398, -0.0102001, -0.0099470, -0.0001520, 0.0001810
8: -0.0046987, -0.0034310, -0.0046596, -0.0032885, -0.0009803, 0.0008232
9: -0.0009946, 0.0053517, -0.0017081, 0.0051564, -0.0041214, 0.0049078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0028843
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0030267
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040152, -0.0036652, -0.0040686, -0.0036765, -0.0002397, 0.0002810
1: 0.0000422, 0.0019801, 0.0001048, 0.0022756, -0.0015560, 0.0013274
2: 0.0105423, 0.0148718, 0.0098822, 0.0147319, -0.0029655, 0.0034763
3: 0.0010674, 0.0028918, 0.0011263, 0.0031700, -0.0014649, 0.0012497
4: 1.0008912, 1.0079695, 1.0011199, 1.0090485, -0.0056833, 0.0048483
5: 0.0024148, 0.0037918, 0.0024593, 0.0040017, -0.0011056, 0.0009432
6: -0.0106774, -0.0088854, -0.0109506, -0.0089433, -0.0012274, 0.0014388
7: -0.0101654, -0.0099368, -0.0102002, -0.0099442, -0.0001566, 0.0001835
8: -0.0047150, -0.0034769, -0.0046750, -0.0032881, -0.0009941, 0.0008480
9: -0.0007648, 0.0054334, -0.0017097, 0.0052332, -0.0042455, 0.0049767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028930, upper bound: 0.0029833
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028930, upper bound: 0.0030716
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040587, -0.0036762, -0.0040228, -0.0036705, -0.0002800, 0.0002269
1: 0.0001028, 0.0022211, 0.0000714, 0.0020224, -0.0012565, 0.0015502
2: 0.0100039, 0.0147364, 0.0104479, 0.0148066, -0.0034632, 0.0028072
3: 0.0011244, 0.0031187, 0.0010948, 0.0029316, -0.0011830, 0.0014594
4: 1.0011126, 1.0088496, 1.0009978, 1.0081236, -0.0045895, 0.0056620
5: 0.0024578, 0.0039630, 0.0024355, 0.0038218, -0.0008928, 0.0011015
6: -0.0109002, -0.0089414, -0.0107165, -0.0089124, -0.0014334, 0.0011619
7: -0.0101938, -0.0099439, -0.0101703, -0.0099402, -0.0001828, 0.0001482
8: -0.0046763, -0.0033229, -0.0046963, -0.0034499, -0.0008028, 0.0009904
9: -0.0015355, 0.0052397, -0.0008999, 0.0053401, -0.0049580, 0.0040189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0028513
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0028513
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040587, -0.0036762, -0.0040790, -0.0036892, -0.0002162, 0.0002337
1: 0.0001028, 0.0022211, 0.0001750, 0.0023333, -0.0012942, 0.0011970
2: 0.0100039, 0.0147364, 0.0097533, 0.0145753, -0.0026741, 0.0028914
3: 0.0011244, 0.0031187, 0.0011923, 0.0032243, -0.0012184, 0.0011269
4: 1.0011126, 1.0088496, 1.0013759, 1.0092593, -0.0047271, 0.0043719
5: 0.0024578, 0.0039630, 0.0025091, 0.0040427, -0.0009196, 0.0008505
6: -0.0109002, -0.0089414, -0.0110040, -0.0090082, -0.0011068, 0.0011967
7: -0.0101938, -0.0099439, -0.0102070, -0.0099524, -0.0001412, 0.0001527
8: -0.0046763, -0.0033229, -0.0046302, -0.0032513, -0.0008268, 0.0007647
9: -0.0015355, 0.0052397, -0.0018943, 0.0050090, -0.0038283, 0.0041394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0028513
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0028513
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040588, -0.0036718, -0.0040097, -0.0036659, -0.0002827, 0.0002326
1: 0.0000786, 0.0022217, 0.0000459, 0.0019496, -0.0012878, 0.0015654
2: 0.0100027, 0.0147906, 0.0106105, 0.0148637, -0.0034972, 0.0028770
3: 0.0011015, 0.0031192, 0.0010708, 0.0028631, -0.0012124, 0.0014737
4: 1.0010238, 1.0088516, 1.0009044, 1.0078579, -0.0047036, 0.0057176
5: 0.0024406, 0.0039634, 0.0024173, 0.0037701, -0.0009150, 0.0011123
6: -0.0109007, -0.0089190, -0.0106491, -0.0088888, -0.0014475, 0.0011908
7: -0.0101938, -0.0099411, -0.0101618, -0.0099372, -0.0001846, 0.0001519
8: -0.0046918, -0.0033226, -0.0047127, -0.0034964, -0.0008227, 0.0010001
9: -0.0015373, 0.0053173, -0.0006671, 0.0054219, -0.0050067, 0.0041188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029535, upper bound: 0.0028245
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029535, upper bound: 0.0028799
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040588, -0.0036718, -0.0040678, -0.0036835, -0.0002157, 0.0002402
1: 0.0000786, 0.0022217, 0.0001436, 0.0022711, -0.0013300, 0.0011944
2: 0.0100027, 0.0147906, 0.0098922, 0.0146454, -0.0026684, 0.0029713
3: 0.0011015, 0.0031192, 0.0011627, 0.0031657, -0.0012521, 0.0011245
4: 1.0010238, 1.0088516, 1.0012612, 1.0090322, -0.0048577, 0.0043625
5: 0.0024406, 0.0039634, 0.0024868, 0.0039985, -0.0009450, 0.0008487
6: -0.0109007, -0.0089190, -0.0109464, -0.0089791, -0.0011044, 0.0012298
7: -0.0101938, -0.0099411, -0.0101997, -0.0099487, -0.0001409, 0.0001569
8: -0.0046918, -0.0033226, -0.0046502, -0.0032910, -0.0008497, 0.0007631
9: -0.0015373, 0.0053173, -0.0016954, 0.0051094, -0.0038201, 0.0042537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029535, upper bound: 0.0028245
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029535, upper bound: 0.0028799
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040685, -0.0036809, -0.0040282, -0.0036698, -0.0002771, 0.0002327
1: 0.0001289, 0.0022751, 0.0000678, 0.0020520, -0.0012886, 0.0015344
2: 0.0098834, 0.0146782, 0.0103817, 0.0148147, -0.0034281, 0.0028788
3: 0.0011489, 0.0031695, 0.0010914, 0.0029595, -0.0012131, 0.0014446
4: 1.0012077, 1.0090467, 1.0009845, 1.0082319, -0.0047065, 0.0056046
5: 0.0024763, 0.0040013, 0.0024329, 0.0038428, -0.0009156, 0.0010903
6: -0.0109501, -0.0089655, -0.0107438, -0.0089091, -0.0014189, 0.0011915
7: -0.0102001, -0.0099470, -0.0101738, -0.0099398, -0.0001810, 0.0001520
8: -0.0046596, -0.0032885, -0.0046987, -0.0034310, -0.0008232, 0.0009803
9: -0.0017081, 0.0051564, -0.0009946, 0.0053517, -0.0049078, 0.0041214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0029225
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0029567
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040685, -0.0036809, -0.0040836, -0.0036884, -0.0002108, 0.0002462
1: 0.0001289, 0.0022751, 0.0001708, 0.0023591, -0.0013634, 0.0011674
2: 0.0098834, 0.0146782, 0.0096957, 0.0145845, -0.0026080, 0.0030460
3: 0.0011489, 0.0031695, 0.0011884, 0.0032486, -0.0012836, 0.0010990
4: 1.0012077, 1.0090467, 1.0013608, 1.0093535, -0.0049799, 0.0042638
5: 0.0024763, 0.0040013, 0.0025061, 0.0040610, -0.0009688, 0.0008295
6: -0.0109501, -0.0089655, -0.0110278, -0.0090043, -0.0010794, 0.0012607
7: -0.0102001, -0.0099470, -0.0102101, -0.0099519, -0.0001377, 0.0001608
8: -0.0046596, -0.0032885, -0.0046328, -0.0032348, -0.0008711, 0.0007458
9: -0.0017081, 0.0051564, -0.0019768, 0.0050222, -0.0037337, 0.0043608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0029225
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0029567
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040686, -0.0036765, -0.0040152, -0.0036652, -0.0002810, 0.0002397
1: 0.0001048, 0.0022756, 0.0000422, 0.0019801, -0.0013274, 0.0015560
2: 0.0098822, 0.0147319, 0.0105423, 0.0148718, -0.0034763, 0.0029655
3: 0.0011263, 0.0031700, 0.0010674, 0.0028918, -0.0012497, 0.0014649
4: 1.0011199, 1.0090485, 1.0008912, 1.0079695, -0.0048483, 0.0056833
5: 0.0024593, 0.0040017, 0.0024148, 0.0037918, -0.0009432, 0.0011056
6: -0.0109506, -0.0089433, -0.0106774, -0.0088854, -0.0014388, 0.0012274
7: -0.0102002, -0.0099442, -0.0101654, -0.0099368, -0.0001835, 0.0001566
8: -0.0046750, -0.0032881, -0.0047150, -0.0034769, -0.0008480, 0.0009941
9: -0.0017097, 0.0052332, -0.0007648, 0.0054334, -0.0049767, 0.0042455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029833, upper bound: 0.0029372
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029833, upper bound: 0.0029503
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040686, -0.0036765, -0.0040724, -0.0036828, -0.0002115, 0.0002541
1: 0.0001048, 0.0022756, 0.0001394, 0.0022968, -0.0014072, 0.0011712
2: 0.0098822, 0.0147319, 0.0098349, 0.0146547, -0.0026166, 0.0031437
3: 0.0011263, 0.0031700, 0.0011588, 0.0031899, -0.0013248, 0.0011026
4: 1.0011199, 1.0090485, 1.0012461, 1.0091258, -0.0051397, 0.0042778
5: 0.0024593, 0.0040017, 0.0024838, 0.0040167, -0.0009999, 0.0008322
6: -0.0109506, -0.0089433, -0.0109701, -0.0089753, -0.0010830, 0.0013012
7: -0.0102002, -0.0099442, -0.0102027, -0.0099482, -0.0001381, 0.0001660
8: -0.0046750, -0.0032881, -0.0046529, -0.0032746, -0.0008990, 0.0007483
9: -0.0017097, 0.0052332, -0.0017774, 0.0051226, -0.0037460, 0.0045007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029833, upper bound: 0.0029372
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029833, upper bound: 0.0029503
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040228, -0.0036705, -0.0040838, -0.0036664, -0.0002431, 0.0003140
1: 0.0000714, 0.0020224, 0.0000486, 0.0023598, -0.0017387, 0.0013462
2: 0.0104479, 0.0148066, 0.0096940, 0.0148577, -0.0030077, 0.0038845
3: 0.0010948, 0.0029316, 0.0010733, 0.0032493, -0.0016369, 0.0012674
4: 1.0009978, 1.0081236, 1.0009142, 1.0093563, -0.0063507, 0.0049172
5: 0.0024355, 0.0038218, 0.0024193, 0.0040616, -0.0012355, 0.0009566
6: -0.0107165, -0.0089124, -0.0110285, -0.0088913, -0.0012449, 0.0016078
7: -0.0101703, -0.0099402, -0.0102101, -0.0099375, -0.0001588, 0.0002051
8: -0.0046963, -0.0034499, -0.0047109, -0.0032343, -0.0011108, 0.0008601
9: -0.0008999, 0.0053401, -0.0019792, 0.0054132, -0.0043058, 0.0055611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0030950
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0031466
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040097, -0.0036659, -0.0040839, -0.0036618, -0.0002498, 0.0003164
1: 0.0000459, 0.0019496, 0.0000233, 0.0023605, -0.0017517, 0.0013832
2: 0.0106105, 0.0148637, 0.0096926, 0.0149141, -0.0030903, 0.0039136
3: 0.0010708, 0.0028631, 0.0010495, 0.0032499, -0.0016492, 0.0013022
4: 1.0009044, 1.0078579, 1.0008221, 1.0093585, -0.0063982, 0.0050522
5: 0.0024173, 0.0037701, 0.0024013, 0.0040620, -0.0012447, 0.0009829
6: -0.0106491, -0.0088888, -0.0110291, -0.0088679, -0.0012790, 0.0016198
7: -0.0101618, -0.0099372, -0.0102102, -0.0099345, -0.0001632, 0.0002066
8: -0.0047127, -0.0034964, -0.0047271, -0.0032339, -0.0011192, 0.0008837
9: -0.0006671, 0.0054219, -0.0019812, 0.0054940, -0.0044241, 0.0056028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028481, upper bound: 0.0031571
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028481, upper bound: 0.0032015
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040282, -0.0036698, -0.0040911, -0.0036756, -0.0002488, 0.0003115
1: 0.0000678, 0.0020520, 0.0000996, 0.0024004, -0.0017248, 0.0013776
2: 0.0103817, 0.0148147, 0.0096033, 0.0147436, -0.0030778, 0.0038534
3: 0.0010914, 0.0029595, 0.0011213, 0.0032875, -0.0016238, 0.0012970
4: 1.0009845, 1.0082319, 1.0011008, 1.0095046, -0.0062999, 0.0050319
5: 0.0024329, 0.0038428, 0.0024555, 0.0040904, -0.0012256, 0.0009789
6: -0.0107438, -0.0089091, -0.0110660, -0.0089385, -0.0012739, 0.0015949
7: -0.0101738, -0.0099398, -0.0102149, -0.0099435, -0.0001625, 0.0002034
8: -0.0046987, -0.0034310, -0.0046783, -0.0032084, -0.0011020, 0.0008802
9: -0.0009946, 0.0053517, -0.0021090, 0.0052500, -0.0044063, 0.0055167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0030950
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0031826
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040152, -0.0036652, -0.0040912, -0.0036708, -0.0002568, 0.0003149
1: 0.0000422, 0.0019801, 0.0000733, 0.0024010, -0.0017433, 0.0014218
2: 0.0105423, 0.0148718, 0.0096020, 0.0148024, -0.0031765, 0.0038948
3: 0.0010674, 0.0028918, 0.0010966, 0.0032880, -0.0016413, 0.0013386
4: 1.0008912, 1.0079695, 1.0010047, 1.0095066, -0.0063676, 0.0051931
5: 0.0024148, 0.0037918, 0.0024369, 0.0040908, -0.0012387, 0.0010103
6: -0.0106774, -0.0088854, -0.0110665, -0.0089142, -0.0013147, 0.0016120
7: -0.0101654, -0.0099368, -0.0102150, -0.0099404, -0.0001677, 0.0002056
8: -0.0047150, -0.0034769, -0.0046951, -0.0032080, -0.0011138, 0.0009084
9: -0.0007648, 0.0054334, -0.0021108, 0.0053341, -0.0045475, 0.0055759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028996, upper bound: 0.0031802
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028996, upper bound: 0.0032137
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040228, -0.0036705, -0.0041423, -0.0036841, -0.0002352, 0.0003798
1: 0.0000714, 0.0020224, 0.0001469, 0.0026836, -0.0021032, 0.0013021
2: 0.0104479, 0.0148066, 0.0089707, 0.0146381, -0.0029090, 0.0046987
3: 0.0010948, 0.0029316, 0.0011658, 0.0035541, -0.0019801, 0.0012259
4: 1.0009978, 1.0081236, 1.0012733, 1.0105388, -0.0076819, 0.0047559
5: 0.0024355, 0.0038218, 0.0024891, 0.0042916, -0.0014944, 0.0009252
6: -0.0107165, -0.0089124, -0.0113278, -0.0089822, -0.0012040, 0.0019448
7: -0.0101703, -0.0099402, -0.0102483, -0.0099491, -0.0001536, 0.0002481
8: -0.0046963, -0.0034499, -0.0046481, -0.0030275, -0.0013437, 0.0008319
9: -0.0008999, 0.0053401, -0.0030147, 0.0050988, -0.0041646, 0.0067268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0027266
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0028134
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040097, -0.0036659, -0.0041424, -0.0036793, -0.0002411, 0.0003829
1: 0.0000459, 0.0019496, 0.0001204, 0.0026841, -0.0021200, 0.0013350
2: 0.0106105, 0.0148637, 0.0089695, 0.0146972, -0.0029825, 0.0047364
3: 0.0010708, 0.0028631, 0.0011409, 0.0035546, -0.0019959, 0.0012568
4: 1.0009044, 1.0078579, 1.0011766, 1.0105407, -0.0077435, 0.0048761
5: 0.0024173, 0.0037701, 0.0024703, 0.0042920, -0.0015064, 0.0009486
6: -0.0106491, -0.0088888, -0.0113284, -0.0089577, -0.0012344, 0.0019604
7: -0.0101618, -0.0099372, -0.0102484, -0.0099460, -0.0001575, 0.0002501
8: -0.0047127, -0.0034964, -0.0046651, -0.0030271, -0.0013545, 0.0008529
9: -0.0006671, 0.0054219, -0.0030164, 0.0051835, -0.0042698, 0.0067808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026186, upper bound: 0.0027528
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026186, upper bound: 0.0030565
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040282, -0.0036698, -0.0041482, -0.0036958, -0.0002365, 0.0003778
1: 0.0000678, 0.0020520, 0.0002116, 0.0027166, -0.0020918, 0.0013097
2: 0.0103817, 0.0148147, 0.0088969, 0.0144933, -0.0029259, 0.0046732
3: 0.0010914, 0.0029595, 0.0012268, 0.0035852, -0.0019693, 0.0012330
4: 1.0009845, 1.0082319, 1.0015100, 1.0106593, -0.0076401, 0.0047835
5: 0.0024329, 0.0038428, 0.0025351, 0.0043151, -0.0014863, 0.0009306
6: -0.0107438, -0.0089091, -0.0113584, -0.0090421, -0.0012110, 0.0019342
7: -0.0101738, -0.0099398, -0.0102522, -0.0099568, -0.0001545, 0.0002467
8: -0.0046987, -0.0034310, -0.0046068, -0.0030064, -0.0013364, 0.0008367
9: -0.0009946, 0.0053517, -0.0031203, 0.0048916, -0.0041888, 0.0066903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0027379
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0029715
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040152, -0.0036652, -0.0041483, -0.0036907, -0.0002441, 0.0003815
1: 0.0000422, 0.0019801, 0.0001833, 0.0027172, -0.0021123, 0.0013518
2: 0.0105423, 0.0148718, 0.0088957, 0.0145566, -0.0030200, 0.0047192
3: 0.0010674, 0.0028918, 0.0012002, 0.0035857, -0.0019887, 0.0012726
4: 1.0008912, 1.0079695, 1.0014064, 1.0106614, -0.0077153, 0.0049373
5: 0.0024148, 0.0037918, 0.0025150, 0.0043155, -0.0015009, 0.0009605
6: -0.0106774, -0.0088854, -0.0113589, -0.0090159, -0.0012500, 0.0019532
7: -0.0101654, -0.0099368, -0.0102523, -0.0099534, -0.0001594, 0.0002492
8: -0.0047150, -0.0034769, -0.0046249, -0.0030060, -0.0013495, 0.0008636
9: -0.0007648, 0.0054334, -0.0031220, 0.0049822, -0.0043235, 0.0067561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028048, upper bound: 0.0030051
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028048, upper bound: 0.0031347
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040587, -0.0036762, -0.0041050, -0.0036837, -0.0002950, 0.0003397
1: 0.0001028, 0.0022211, 0.0001443, 0.0024772, -0.0018808, 0.0016337
2: 0.0100039, 0.0147364, 0.0094318, 0.0146437, -0.0036498, 0.0042019
3: 0.0011244, 0.0031187, 0.0011635, 0.0033598, -0.0017707, 0.0015380
4: 1.0011126, 1.0088496, 1.0012641, 1.0097851, -0.0068695, 0.0059670
5: 0.0024578, 0.0039630, 0.0024873, 0.0041450, -0.0013364, 0.0011608
6: -0.0109002, -0.0089414, -0.0111370, -0.0089798, -0.0015106, 0.0017391
7: -0.0101938, -0.0099439, -0.0102240, -0.0099488, -0.0001927, 0.0002218
8: -0.0046763, -0.0033229, -0.0046497, -0.0031593, -0.0012016, 0.0010437
9: -0.0015355, 0.0052397, -0.0023546, 0.0051069, -0.0052252, 0.0060155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025799, upper bound: 0.0026477
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025799, upper bound: 0.0026782
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040587, -0.0036762, -0.0041610, -0.0037048, -0.0002298, 0.0003461
1: 0.0001028, 0.0022211, 0.0002614, 0.0027874, -0.0019164, 0.0012725
2: 0.0100039, 0.0147364, 0.0087387, 0.0143821, -0.0028429, 0.0042815
3: 0.0011244, 0.0031187, 0.0012737, 0.0036518, -0.0018042, 0.0011980
4: 1.0011126, 1.0088496, 1.0016918, 1.0109180, -0.0069997, 0.0046479
5: 0.0024578, 0.0039630, 0.0025705, 0.0043654, -0.0013617, 0.0009042
6: -0.0109002, -0.0089414, -0.0114239, -0.0090881, -0.0011767, 0.0017721
7: -0.0101938, -0.0099439, -0.0102606, -0.0099626, -0.0001501, 0.0002260
8: -0.0046763, -0.0033229, -0.0045749, -0.0029611, -0.0012244, 0.0008130
9: -0.0015355, 0.0052397, -0.0033468, 0.0047324, -0.0040700, 0.0061295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025799, upper bound: 0.0026477
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025799, upper bound: 0.0026782
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040588, -0.0036718, -0.0040916, -0.0036775, -0.0002983, 0.0003419
1: 0.0000786, 0.0022217, 0.0001100, 0.0024033, -0.0018932, 0.0016516
2: 0.0100027, 0.0147906, 0.0095969, 0.0147203, -0.0036898, 0.0042296
3: 0.0011015, 0.0031192, 0.0011312, 0.0032902, -0.0017823, 0.0015549
4: 1.0010238, 1.0088516, 1.0011388, 1.0095149, -0.0069148, 0.0060323
5: 0.0024406, 0.0039634, 0.0024629, 0.0040924, -0.0013452, 0.0011735
6: -0.0109007, -0.0089190, -0.0110687, -0.0089481, -0.0015272, 0.0017506
7: -0.0101938, -0.0099411, -0.0102153, -0.0099448, -0.0001948, 0.0002233
8: -0.0046918, -0.0033226, -0.0046717, -0.0032066, -0.0012095, 0.0010551
9: -0.0015373, 0.0053173, -0.0021182, 0.0052166, -0.0052823, 0.0060551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027054, upper bound: 0.0026419
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027054, upper bound: 0.0029787
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040588, -0.0036718, -0.0041497, -0.0036972, -0.0002299, 0.0003496
1: 0.0000786, 0.0022217, 0.0002196, 0.0027247, -0.0019357, 0.0012727
2: 0.0100027, 0.0147906, 0.0088788, 0.0144756, -0.0028433, 0.0043246
3: 0.0011015, 0.0031192, 0.0012343, 0.0035928, -0.0018224, 0.0011982
4: 1.0010238, 1.0088516, 1.0015389, 1.0106890, -0.0070702, 0.0046485
5: 0.0024406, 0.0039634, 0.0025408, 0.0043208, -0.0013754, 0.0009043
6: -0.0109007, -0.0089190, -0.0113659, -0.0090494, -0.0011768, 0.0017899
7: -0.0101938, -0.0099411, -0.0102532, -0.0099577, -0.0001501, 0.0002283
8: -0.0046918, -0.0033226, -0.0046017, -0.0030012, -0.0012367, 0.0008131
9: -0.0015373, 0.0053173, -0.0031463, 0.0048663, -0.0040705, 0.0061912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027054, upper bound: 0.0026419
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027054, upper bound: 0.0029787
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040685, -0.0036809, -0.0041090, -0.0036829, -0.0002986, 0.0003427
1: 0.0001289, 0.0022751, 0.0001400, 0.0024996, -0.0018978, 0.0016532
2: 0.0098834, 0.0146782, 0.0093817, 0.0146534, -0.0036935, 0.0042399
3: 0.0011489, 0.0031695, 0.0011594, 0.0033809, -0.0017867, 0.0015565
4: 1.0012077, 1.0090467, 1.0012481, 1.0098668, -0.0069317, 0.0060385
5: 0.0024763, 0.0040013, 0.0024842, 0.0041609, -0.0013485, 0.0011747
6: -0.0109501, -0.0089655, -0.0111578, -0.0089758, -0.0015287, 0.0017549
7: -0.0102001, -0.0099470, -0.0102266, -0.0099483, -0.0001950, 0.0002238
8: -0.0046596, -0.0032885, -0.0046525, -0.0031450, -0.0012125, 0.0010562
9: -0.0017081, 0.0051564, -0.0024263, 0.0051208, -0.0052877, 0.0060699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025801, upper bound: 0.0027371
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025801, upper bound: 0.0029471
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040685, -0.0036809, -0.0041641, -0.0037039, -0.0002315, 0.0003557
1: 0.0001289, 0.0022751, 0.0002563, 0.0028046, -0.0019697, 0.0012819
2: 0.0098834, 0.0146782, 0.0087004, 0.0143935, -0.0028639, 0.0044005
3: 0.0011489, 0.0031695, 0.0012689, 0.0036680, -0.0018544, 0.0012069
4: 1.0012077, 1.0090467, 1.0016732, 1.0109807, -0.0071943, 0.0046821
5: 0.0024763, 0.0040013, 0.0025669, 0.0043776, -0.0013996, 0.0009109
6: -0.0109501, -0.0089655, -0.0114397, -0.0090834, -0.0011853, 0.0018213
7: -0.0102001, -0.0099470, -0.0102626, -0.0099620, -0.0001512, 0.0002323
8: -0.0046596, -0.0032885, -0.0045782, -0.0029502, -0.0012584, 0.0008190
9: -0.0017081, 0.0051564, -0.0034017, 0.0047487, -0.0041000, 0.0062998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025801, upper bound: 0.0027371
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025801, upper bound: 0.0029471
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040686, -0.0036765, -0.0040958, -0.0036767, -0.0003029, 0.0003467
1: 0.0001048, 0.0022756, 0.0001060, 0.0024263, -0.0019194, 0.0016772
2: 0.0098822, 0.0147319, 0.0095455, 0.0147292, -0.0037470, 0.0042881
3: 0.0011263, 0.0031700, 0.0011274, 0.0033118, -0.0018070, 0.0015790
4: 1.0011199, 1.0090485, 1.0011241, 1.0095990, -0.0070106, 0.0061259
5: 0.0024593, 0.0040017, 0.0024601, 0.0041088, -0.0013638, 0.0011917
6: -0.0109506, -0.0089433, -0.0110899, -0.0089444, -0.0015509, 0.0017748
7: -0.0102002, -0.0099442, -0.0102180, -0.0099443, -0.0001978, 0.0002264
8: -0.0046750, -0.0032881, -0.0046742, -0.0031919, -0.0012263, 0.0010715
9: -0.0017097, 0.0052332, -0.0021917, 0.0052294, -0.0053643, 0.0061390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028657, upper bound: 0.0030366
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028657, upper bound: 0.0030876
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040686, -0.0036765, -0.0041533, -0.0036964, -0.0002326, 0.0003611
1: 0.0001048, 0.0022756, 0.0002147, 0.0027445, -0.0019996, 0.0012878
2: 0.0098822, 0.0147319, 0.0088346, 0.0144865, -0.0028770, 0.0044674
3: 0.0011263, 0.0031700, 0.0012297, 0.0036114, -0.0018826, 0.0012124
4: 1.0011199, 1.0090485, 1.0015212, 1.0107613, -0.0073037, 0.0047035
5: 0.0024593, 0.0040017, 0.0025373, 0.0043349, -0.0014209, 0.0009150
6: -0.0109506, -0.0089433, -0.0113842, -0.0090449, -0.0011908, 0.0018490
7: -0.0102002, -0.0099442, -0.0102555, -0.0099571, -0.0001519, 0.0002359
8: -0.0046750, -0.0032881, -0.0046048, -0.0029885, -0.0012775, 0.0008227
9: -0.0017097, 0.0052332, -0.0032095, 0.0048818, -0.0041188, 0.0063956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028657, upper bound: 0.0030366
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028657, upper bound: 0.0030876
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040838, -0.0036664, -0.0040228, -0.0036705, -0.0003140, 0.0002431
1: 0.0000486, 0.0023598, 0.0000714, 0.0020224, -0.0013462, 0.0017387
2: 0.0096940, 0.0148577, 0.0104479, 0.0148066, -0.0038845, 0.0030077
3: 0.0010733, 0.0032493, 0.0010948, 0.0029316, -0.0012674, 0.0016369
4: 1.0009142, 1.0093563, 1.0009978, 1.0081236, -0.0049172, 0.0063507
5: 0.0024193, 0.0040616, 0.0024355, 0.0038218, -0.0009566, 0.0012355
6: -0.0110285, -0.0088913, -0.0107165, -0.0089124, -0.0016078, 0.0012449
7: -0.0102101, -0.0099375, -0.0101703, -0.0099402, -0.0002051, 0.0001588
8: -0.0047109, -0.0032343, -0.0046963, -0.0034499, -0.0008601, 0.0011108
9: -0.0019792, 0.0054132, -0.0008999, 0.0053401, -0.0055611, 0.0043058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0028824
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0028824
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0036618, -0.0040097, -0.0036659, -0.0003164, 0.0002498
1: 0.0000233, 0.0023605, 0.0000459, 0.0019496, -0.0013832, 0.0017517
2: 0.0096926, 0.0149141, 0.0106105, 0.0148637, -0.0039136, 0.0030903
3: 0.0010495, 0.0032499, 0.0010708, 0.0028631, -0.0013022, 0.0016492
4: 1.0008221, 1.0093585, 1.0009044, 1.0078579, -0.0050522, 0.0063982
5: 0.0024013, 0.0040620, 0.0024173, 0.0037701, -0.0009829, 0.0012447
6: -0.0110291, -0.0088679, -0.0106491, -0.0088888, -0.0016198, 0.0012790
7: -0.0102102, -0.0099345, -0.0101618, -0.0099372, -0.0002066, 0.0001632
8: -0.0047271, -0.0032339, -0.0047127, -0.0034964, -0.0008837, 0.0011192
9: -0.0019812, 0.0054940, -0.0006671, 0.0054219, -0.0056028, 0.0044241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031571, upper bound: 0.0028481
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031571, upper bound: 0.0028996
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040911, -0.0036756, -0.0040282, -0.0036698, -0.0003115, 0.0002488
1: 0.0000996, 0.0024004, 0.0000678, 0.0020520, -0.0013776, 0.0017248
2: 0.0096033, 0.0147436, 0.0103817, 0.0148147, -0.0038534, 0.0030778
3: 0.0011213, 0.0032875, 0.0010914, 0.0029595, -0.0012970, 0.0016238
4: 1.0011008, 1.0095046, 1.0009845, 1.0082319, -0.0050319, 0.0062999
5: 0.0024555, 0.0040904, 0.0024329, 0.0038428, -0.0009789, 0.0012256
6: -0.0110660, -0.0089385, -0.0107438, -0.0089091, -0.0015949, 0.0012739
7: -0.0102149, -0.0099435, -0.0101738, -0.0099398, -0.0002034, 0.0001625
8: -0.0046783, -0.0032084, -0.0046987, -0.0034310, -0.0008802, 0.0011020
9: -0.0021090, 0.0052500, -0.0009946, 0.0053517, -0.0055167, 0.0044063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0029455
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0029657
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040912, -0.0036708, -0.0040152, -0.0036652, -0.0003149, 0.0002568
1: 0.0000733, 0.0024010, 0.0000422, 0.0019801, -0.0014218, 0.0017433
2: 0.0096020, 0.0148024, 0.0105423, 0.0148718, -0.0038948, 0.0031765
3: 0.0010966, 0.0032880, 0.0010674, 0.0028918, -0.0013386, 0.0016413
4: 1.0010047, 1.0095066, 1.0008912, 1.0079695, -0.0051931, 0.0063676
5: 0.0024369, 0.0040908, 0.0024148, 0.0037918, -0.0010103, 0.0012387
6: -0.0110665, -0.0089142, -0.0106774, -0.0088854, -0.0016120, 0.0013147
7: -0.0102150, -0.0099404, -0.0101654, -0.0099368, -0.0002056, 0.0001677
8: -0.0046951, -0.0032080, -0.0047150, -0.0034769, -0.0009084, 0.0011138
9: -0.0021108, 0.0053341, -0.0007648, 0.0054334, -0.0055759, 0.0045475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031802, upper bound: 0.0029563
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031802, upper bound: 0.0029608
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041423, -0.0036841, -0.0040228, -0.0036705, -0.0003798, 0.0002352
1: 0.0001469, 0.0026836, 0.0000714, 0.0020224, -0.0013021, 0.0021032
2: 0.0089707, 0.0146381, 0.0104479, 0.0148066, -0.0046987, 0.0029090
3: 0.0011658, 0.0035541, 0.0010948, 0.0029316, -0.0012259, 0.0019801
4: 1.0012733, 1.0105388, 1.0009978, 1.0081236, -0.0047559, 0.0076819
5: 0.0024891, 0.0042916, 0.0024355, 0.0038218, -0.0009252, 0.0014944
6: -0.0113278, -0.0089822, -0.0107165, -0.0089124, -0.0019448, 0.0012040
7: -0.0102483, -0.0099491, -0.0101703, -0.0099402, -0.0002481, 0.0001536
8: -0.0046481, -0.0030275, -0.0046963, -0.0034499, -0.0008319, 0.0013437
9: -0.0030147, 0.0050988, -0.0008999, 0.0053401, -0.0067268, 0.0041646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027266, upper bound: 0.0027041
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027266, upper bound: 0.0027041
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041424, -0.0036793, -0.0040097, -0.0036659, -0.0003829, 0.0002411
1: 0.0001204, 0.0026841, 0.0000459, 0.0019496, -0.0013350, 0.0021200
2: 0.0089695, 0.0146972, 0.0106105, 0.0148637, -0.0047364, 0.0029825
3: 0.0011409, 0.0035546, 0.0010708, 0.0028631, -0.0012568, 0.0019959
4: 1.0011766, 1.0105407, 1.0009044, 1.0078579, -0.0048761, 0.0077435
5: 0.0024703, 0.0042920, 0.0024173, 0.0037701, -0.0009486, 0.0015064
6: -0.0113284, -0.0089577, -0.0106491, -0.0088888, -0.0019604, 0.0012344
7: -0.0102484, -0.0099460, -0.0101618, -0.0099372, -0.0002501, 0.0001575
8: -0.0046651, -0.0030271, -0.0047127, -0.0034964, -0.0008529, 0.0013545
9: -0.0030164, 0.0051835, -0.0006671, 0.0054219, -0.0067808, 0.0042698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027528, upper bound: 0.0026186
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027528, upper bound: 0.0028048
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041482, -0.0036958, -0.0040282, -0.0036698, -0.0003778, 0.0002365
1: 0.0002116, 0.0027166, 0.0000678, 0.0020520, -0.0013097, 0.0020918
2: 0.0088969, 0.0144933, 0.0103817, 0.0148147, -0.0046732, 0.0029259
3: 0.0012268, 0.0035852, 0.0010914, 0.0029595, -0.0012330, 0.0019693
4: 1.0015100, 1.0106593, 1.0009845, 1.0082319, -0.0047835, 0.0076401
5: 0.0025351, 0.0043151, 0.0024329, 0.0038428, -0.0009306, 0.0014863
6: -0.0113584, -0.0090421, -0.0107438, -0.0089091, -0.0019342, 0.0012110
7: -0.0102522, -0.0099568, -0.0101738, -0.0099398, -0.0002467, 0.0001545
8: -0.0046068, -0.0030064, -0.0046987, -0.0034310, -0.0008367, 0.0013364
9: -0.0031203, 0.0048916, -0.0009946, 0.0053517, -0.0066903, 0.0041888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027379, upper bound: 0.0028021
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027379, upper bound: 0.0028416
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041483, -0.0036907, -0.0040152, -0.0036652, -0.0003815, 0.0002441
1: 0.0001833, 0.0027172, 0.0000422, 0.0019801, -0.0013518, 0.0021123
2: 0.0088957, 0.0145566, 0.0105423, 0.0148718, -0.0047192, 0.0030200
3: 0.0012002, 0.0035857, 0.0010674, 0.0028918, -0.0012726, 0.0019887
4: 1.0014064, 1.0106614, 1.0008912, 1.0079695, -0.0049373, 0.0077153
5: 0.0025150, 0.0043155, 0.0024148, 0.0037918, -0.0009605, 0.0015009
6: -0.0113589, -0.0090159, -0.0106774, -0.0088854, -0.0019532, 0.0012500
7: -0.0102523, -0.0099534, -0.0101654, -0.0099368, -0.0002492, 0.0001594
8: -0.0046249, -0.0030060, -0.0047150, -0.0034769, -0.0008636, 0.0013495
9: -0.0031220, 0.0049822, -0.0007648, 0.0054334, -0.0067561, 0.0043235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030051, upper bound: 0.0028783
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030051, upper bound: 0.0028955
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040838, -0.0036664, -0.0041050, -0.0036837, -0.0002251, 0.0002425
1: 0.0000486, 0.0023598, 0.0001446, 0.0024772, -0.0013427, 0.0012462
2: 0.0096940, 0.0148577, 0.0094318, 0.0146432, -0.0027842, 0.0029998
3: 0.0010733, 0.0032493, 0.0011637, 0.0033598, -0.0012641, 0.0011733
4: 1.0009142, 1.0093563, 1.0012649, 1.0097851, -0.0049043, 0.0045518
5: 0.0024193, 0.0040616, 0.0024875, 0.0041450, -0.0009541, 0.0008855
6: -0.0110285, -0.0088913, -0.0111370, -0.0089800, -0.0011523, 0.0012416
7: -0.0102101, -0.0099375, -0.0102240, -0.0099488, -0.0001470, 0.0001584
8: -0.0047109, -0.0032343, -0.0046496, -0.0031593, -0.0008578, 0.0007962
9: -0.0019792, 0.0054132, -0.0023546, 0.0051062, -0.0039859, 0.0042946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0028783
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0028783
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0036618, -0.0040916, -0.0036775, -0.0002246, 0.0002501
1: 0.0000233, 0.0023605, 0.0001101, 0.0024033, -0.0013846, 0.0012439
2: 0.0096926, 0.0149141, 0.0095969, 0.0147201, -0.0027789, 0.0030933
3: 0.0010495, 0.0032499, 0.0011313, 0.0032902, -0.0013035, 0.0011710
4: 1.0008221, 1.0093585, 1.0011393, 1.0095149, -0.0050572, 0.0045432
5: 0.0024013, 0.0040620, 0.0024630, 0.0040924, -0.0009838, 0.0008838
6: -0.0110291, -0.0088679, -0.0110687, -0.0089482, -0.0011502, 0.0012803
7: -0.0102102, -0.0099345, -0.0102153, -0.0099448, -0.0001467, 0.0001633
8: -0.0047271, -0.0032339, -0.0046716, -0.0032066, -0.0008846, 0.0007947
9: -0.0019812, 0.0054940, -0.0021182, 0.0052163, -0.0039784, 0.0044284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031571, upper bound: 0.0028481
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031571, upper bound: 0.0028996
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040911, -0.0036756, -0.0041090, -0.0036829, -0.0002196, 0.0002560
1: 0.0000996, 0.0024004, 0.0001402, 0.0024996, -0.0014173, 0.0012159
2: 0.0096033, 0.0147436, 0.0093817, 0.0146529, -0.0027164, 0.0031663
3: 0.0011213, 0.0032875, 0.0011596, 0.0033809, -0.0013343, 0.0011447
4: 1.0011008, 1.0095046, 1.0012490, 1.0098668, -0.0051766, 0.0044410
5: 0.0024555, 0.0040904, 0.0024844, 0.0041609, -0.0010070, 0.0008640
6: -0.0110660, -0.0089385, -0.0111578, -0.0089760, -0.0011243, 0.0013105
7: -0.0102149, -0.0099435, -0.0102266, -0.0099483, -0.0001434, 0.0001672
8: -0.0046783, -0.0032084, -0.0046524, -0.0031450, -0.0009055, 0.0007768
9: -0.0021090, 0.0052500, -0.0024263, 0.0051201, -0.0038889, 0.0045330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0029437
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0029644
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040912, -0.0036708, -0.0040958, -0.0036768, -0.0002200, 0.0002645
1: 0.0000733, 0.0024010, 0.0001062, 0.0024263, -0.0014648, 0.0012179
2: 0.0096020, 0.0148024, 0.0095455, 0.0147290, -0.0027209, 0.0032724
3: 0.0010966, 0.0032880, 0.0011275, 0.0033118, -0.0013790, 0.0011466
4: 1.0010047, 1.0095066, 1.0011247, 1.0095990, -0.0053500, 0.0044483
5: 0.0024369, 0.0040908, 0.0024602, 0.0041088, -0.0010408, 0.0008654
6: -0.0110665, -0.0089142, -0.0110899, -0.0089445, -0.0011262, 0.0013544
7: -0.0102150, -0.0099404, -0.0102180, -0.0099443, -0.0001437, 0.0001728
8: -0.0046951, -0.0032080, -0.0046741, -0.0031919, -0.0009358, 0.0007781
9: -0.0021108, 0.0053341, -0.0021917, 0.0052290, -0.0038953, 0.0046849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031802, upper bound: 0.0029562
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031802, upper bound: 0.0029608
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041423, -0.0036841, -0.0041050, -0.0036837, -0.0003007, 0.0002476
1: 0.0001469, 0.0026836, 0.0001446, 0.0024772, -0.0013709, 0.0016647
2: 0.0089707, 0.0146381, 0.0094318, 0.0146432, -0.0037192, 0.0030628
3: 0.0011658, 0.0035541, 0.0011637, 0.0033598, -0.0012907, 0.0015673
4: 1.0012733, 1.0105388, 1.0012649, 1.0097851, -0.0050074, 0.0060805
5: 0.0024891, 0.0042916, 0.0024875, 0.0041450, -0.0009741, 0.0011829
6: -0.0113278, -0.0089822, -0.0111370, -0.0089800, -0.0015394, 0.0012677
7: -0.0102483, -0.0099491, -0.0102240, -0.0099488, -0.0001964, 0.0001617
8: -0.0046481, -0.0030275, -0.0046496, -0.0031593, -0.0008759, 0.0010636
9: -0.0030147, 0.0050988, -0.0023546, 0.0051062, -0.0053245, 0.0043848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027122, upper bound: 0.0026859
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027122, upper bound: 0.0026859
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041424, -0.0036793, -0.0040916, -0.0036775, -0.0003037, 0.0002534
1: 0.0001204, 0.0026841, 0.0001101, 0.0024033, -0.0014031, 0.0016813
2: 0.0089695, 0.0146972, 0.0095969, 0.0147201, -0.0037563, 0.0031347
3: 0.0011409, 0.0035546, 0.0011313, 0.0032902, -0.0013210, 0.0015829
4: 1.0011766, 1.0105407, 1.0011393, 1.0095149, -0.0051249, 0.0061410
5: 0.0024703, 0.0042920, 0.0024630, 0.0040924, -0.0009970, 0.0011947
6: -0.0113284, -0.0089577, -0.0110687, -0.0089482, -0.0015547, 0.0012975
7: -0.0102484, -0.0099460, -0.0102153, -0.0099448, -0.0001983, 0.0001655
8: -0.0046651, -0.0030271, -0.0046716, -0.0032066, -0.0008964, 0.0010742
9: -0.0030164, 0.0051835, -0.0021182, 0.0052163, -0.0053776, 0.0044878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027206, upper bound: 0.0026049
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027206, upper bound: 0.0028031
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041482, -0.0036958, -0.0041090, -0.0036829, -0.0002962, 0.0002535
1: 0.0002116, 0.0027166, 0.0001402, 0.0024996, -0.0014038, 0.0016401
2: 0.0088969, 0.0144933, 0.0093817, 0.0146529, -0.0036641, 0.0031362
3: 0.0012268, 0.0035852, 0.0011596, 0.0033809, -0.0013216, 0.0015441
4: 1.0015100, 1.0106593, 1.0012490, 1.0098668, -0.0051273, 0.0059903
5: 0.0025351, 0.0043151, 0.0024844, 0.0041609, -0.0009975, 0.0011654
6: -0.0113584, -0.0090421, -0.0111578, -0.0089760, -0.0015165, 0.0012981
7: -0.0102522, -0.0099568, -0.0102266, -0.0099483, -0.0001934, 0.0001656
8: -0.0046068, -0.0030064, -0.0046524, -0.0031450, -0.0008968, 0.0010478
9: -0.0031203, 0.0048916, -0.0024263, 0.0051201, -0.0052456, 0.0044898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027279, upper bound: 0.0027853
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027279, upper bound: 0.0028346
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041483, -0.0036907, -0.0040958, -0.0036768, -0.0003001, 0.0002610
1: 0.0001833, 0.0027172, 0.0001062, 0.0024263, -0.0014449, 0.0016614
2: 0.0088957, 0.0145566, 0.0095455, 0.0147290, -0.0037117, 0.0032281
3: 0.0012002, 0.0035857, 0.0011275, 0.0033118, -0.0013603, 0.0015641
4: 1.0014064, 1.0106614, 1.0011247, 1.0095990, -0.0052776, 0.0060682
5: 0.0025150, 0.0043155, 0.0024602, 0.0041088, -0.0010267, 0.0011805
6: -0.0113589, -0.0090159, -0.0110899, -0.0089445, -0.0015362, 0.0013361
7: -0.0102523, -0.0099534, -0.0102180, -0.0099443, -0.0001960, 0.0001704
8: -0.0046249, -0.0030060, -0.0046741, -0.0031919, -0.0009231, 0.0010614
9: -0.0031220, 0.0049822, -0.0021917, 0.0052290, -0.0053137, 0.0046215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030051, upper bound: 0.0028782
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030051, upper bound: 0.0028955
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0036837, -0.0040587, -0.0036762, -0.0003397, 0.0002950
1: 0.0001443, 0.0024772, 0.0001028, 0.0022211, -0.0016337, 0.0018808
2: 0.0094318, 0.0146437, 0.0100039, 0.0147364, -0.0042019, 0.0036498
3: 0.0011635, 0.0033598, 0.0011244, 0.0031187, -0.0015380, 0.0017707
4: 1.0012641, 1.0097851, 1.0011126, 1.0088496, -0.0059670, 0.0068695
5: 0.0024873, 0.0041450, 0.0024578, 0.0039630, -0.0011608, 0.0013364
6: -0.0111370, -0.0089798, -0.0109002, -0.0089414, -0.0017391, 0.0015106
7: -0.0102240, -0.0099488, -0.0101938, -0.0099439, -0.0002218, 0.0001927
8: -0.0046497, -0.0031593, -0.0046763, -0.0033229, -0.0010437, 0.0012016
9: -0.0023546, 0.0051069, -0.0015355, 0.0052397, -0.0060155, 0.0052252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0025799
time: 0.70 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0027161
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041610, -0.0037048, -0.0040587, -0.0036762, -0.0003461, 0.0002298
1: 0.0002614, 0.0027874, 0.0001028, 0.0022211, -0.0012725, 0.0019164
2: 0.0087387, 0.0143821, 0.0100039, 0.0147364, -0.0042815, 0.0028429
3: 0.0012737, 0.0036518, 0.0011244, 0.0031187, -0.0011980, 0.0018042
4: 1.0016918, 1.0109180, 1.0011126, 1.0088496, -0.0046479, 0.0069997
5: 0.0025705, 0.0043654, 0.0024578, 0.0039630, -0.0009042, 0.0013617
6: -0.0114239, -0.0090881, -0.0109002, -0.0089414, -0.0017721, 0.0011767
7: -0.0102606, -0.0099626, -0.0101938, -0.0099439, -0.0002260, 0.0001501
8: -0.0045749, -0.0029611, -0.0046763, -0.0033229, -0.0008130, 0.0012244
9: -0.0033468, 0.0047324, -0.0015355, 0.0052397, -0.0061295, 0.0040700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0025718
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0027080
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040916, -0.0036775, -0.0040588, -0.0036718, -0.0003419, 0.0002983
1: 0.0001100, 0.0024033, 0.0000786, 0.0022217, -0.0016516, 0.0018932
2: 0.0095969, 0.0147203, 0.0100027, 0.0147906, -0.0042296, 0.0036898
3: 0.0011312, 0.0032902, 0.0011015, 0.0031192, -0.0015549, 0.0017823
4: 1.0011388, 1.0095149, 1.0010238, 1.0088516, -0.0060323, 0.0069148
5: 0.0024629, 0.0040924, 0.0024406, 0.0039634, -0.0011735, 0.0013452
6: -0.0110687, -0.0089481, -0.0109007, -0.0089190, -0.0017506, 0.0015272
7: -0.0102153, -0.0099448, -0.0101938, -0.0099411, -0.0002233, 0.0001948
8: -0.0046717, -0.0032066, -0.0046918, -0.0033226, -0.0010551, 0.0012095
9: -0.0021182, 0.0052166, -0.0015373, 0.0053173, -0.0060551, 0.0052823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026419, upper bound: 0.0027547
time: 0.65 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026419, upper bound: 0.0029203
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041497, -0.0036972, -0.0040588, -0.0036718, -0.0003496, 0.0002299
1: 0.0002196, 0.0027247, 0.0000786, 0.0022217, -0.0012727, 0.0019357
2: 0.0088788, 0.0144756, 0.0100027, 0.0147906, -0.0043246, 0.0028433
3: 0.0012343, 0.0035928, 0.0011015, 0.0031192, -0.0011982, 0.0018224
4: 1.0015389, 1.0106890, 1.0010238, 1.0088516, -0.0046485, 0.0070702
5: 0.0025408, 0.0043208, 0.0024406, 0.0039634, -0.0009043, 0.0013754
6: -0.0113659, -0.0090494, -0.0109007, -0.0089190, -0.0017899, 0.0011768
7: -0.0102532, -0.0099577, -0.0101938, -0.0099411, -0.0002283, 0.0001501
8: -0.0046017, -0.0030012, -0.0046918, -0.0033226, -0.0008131, 0.0012367
9: -0.0031463, 0.0048663, -0.0015373, 0.0053173, -0.0061912, 0.0040705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026419, upper bound: 0.0027511
time: 0.81 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026419, upper bound: 0.0028601
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041090, -0.0036829, -0.0040685, -0.0036809, -0.0003427, 0.0002986
1: 0.0001400, 0.0024996, 0.0001289, 0.0022751, -0.0016532, 0.0018978
2: 0.0093817, 0.0146534, 0.0098834, 0.0146782, -0.0042399, 0.0036935
3: 0.0011594, 0.0033809, 0.0011489, 0.0031695, -0.0015565, 0.0017867
4: 1.0012481, 1.0098668, 1.0012077, 1.0090467, -0.0060385, 0.0069317
5: 0.0024842, 0.0041609, 0.0024763, 0.0040013, -0.0011747, 0.0013485
6: -0.0111578, -0.0089758, -0.0109501, -0.0089655, -0.0017549, 0.0015287
7: -0.0102266, -0.0099483, -0.0102001, -0.0099470, -0.0002238, 0.0001950
8: -0.0046525, -0.0031450, -0.0046596, -0.0032885, -0.0010562, 0.0012125
9: -0.0024263, 0.0051208, -0.0017081, 0.0051564, -0.0060699, 0.0052877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0025801
time: 0.83 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0028432
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041641, -0.0037039, -0.0040685, -0.0036809, -0.0003557, 0.0002315
1: 0.0002563, 0.0028046, 0.0001289, 0.0022751, -0.0012819, 0.0019697
2: 0.0087004, 0.0143935, 0.0098834, 0.0146782, -0.0044005, 0.0028639
3: 0.0012689, 0.0036680, 0.0011489, 0.0031695, -0.0012069, 0.0018544
4: 1.0016732, 1.0109807, 1.0012077, 1.0090467, -0.0046821, 0.0071943
5: 0.0025669, 0.0043776, 0.0024763, 0.0040013, -0.0009109, 0.0013996
6: -0.0114397, -0.0090834, -0.0109501, -0.0089655, -0.0018213, 0.0011853
7: -0.0102626, -0.0099620, -0.0102001, -0.0099470, -0.0002323, 0.0001512
8: -0.0045782, -0.0029502, -0.0046596, -0.0032885, -0.0008190, 0.0012584
9: -0.0034017, 0.0047487, -0.0017081, 0.0051564, -0.0062998, 0.0041000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0025718
time: 0.87 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0027877
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040958, -0.0036767, -0.0040686, -0.0036765, -0.0003467, 0.0003029
1: 0.0001060, 0.0024263, 0.0001048, 0.0022756, -0.0016772, 0.0019194
2: 0.0095455, 0.0147292, 0.0098822, 0.0147319, -0.0042881, 0.0037470
3: 0.0011274, 0.0033118, 0.0011263, 0.0031700, -0.0015790, 0.0018070
4: 1.0011241, 1.0095990, 1.0011199, 1.0090485, -0.0061259, 0.0070106
5: 0.0024601, 0.0041088, 0.0024593, 0.0040017, -0.0011917, 0.0013638
6: -0.0110899, -0.0089444, -0.0109506, -0.0089433, -0.0017748, 0.0015509
7: -0.0102180, -0.0099443, -0.0102002, -0.0099442, -0.0002264, 0.0001978
8: -0.0046742, -0.0031919, -0.0046750, -0.0032881, -0.0010715, 0.0012263
9: -0.0021917, 0.0052294, -0.0017097, 0.0052332, -0.0061390, 0.0053643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028657
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0029659
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041533, -0.0036964, -0.0040686, -0.0036765, -0.0003611, 0.0002326
1: 0.0002147, 0.0027445, 0.0001048, 0.0022756, -0.0012878, 0.0019996
2: 0.0088346, 0.0144865, 0.0098822, 0.0147319, -0.0044674, 0.0028770
3: 0.0012297, 0.0036114, 0.0011263, 0.0031700, -0.0012124, 0.0018826
4: 1.0015212, 1.0107613, 1.0011199, 1.0090485, -0.0047035, 0.0073037
5: 0.0025373, 0.0043349, 0.0024593, 0.0040017, -0.0009150, 0.0014209
6: -0.0113842, -0.0090449, -0.0109506, -0.0089433, -0.0018490, 0.0011908
7: -0.0102555, -0.0099571, -0.0102002, -0.0099442, -0.0002359, 0.0001519
8: -0.0046048, -0.0029885, -0.0046750, -0.0032881, -0.0008227, 0.0012775
9: -0.0032095, 0.0048818, -0.0017097, 0.0052332, -0.0063956, 0.0041188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0027911
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028811
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0036837, -0.0041423, -0.0036841, -0.0002476, 0.0003012
1: 0.0001443, 0.0024772, 0.0001469, 0.0026836, -0.0016677, 0.0013709
2: 0.0094318, 0.0146437, 0.0089707, 0.0146381, -0.0030628, 0.0037258
3: 0.0011635, 0.0033598, 0.0011658, 0.0035541, -0.0015701, 0.0012907
4: 1.0012641, 1.0097851, 1.0012733, 1.0105388, -0.0060912, 0.0050074
5: 0.0024873, 0.0041450, 0.0024891, 0.0042916, -0.0011850, 0.0009741
6: -0.0111370, -0.0089798, -0.0113278, -0.0089822, -0.0012677, 0.0015421
7: -0.0102240, -0.0099488, -0.0102483, -0.0099491, -0.0001617, 0.0001967
8: -0.0046497, -0.0031593, -0.0046481, -0.0030275, -0.0010655, 0.0008759
9: -0.0023546, 0.0051069, -0.0030147, 0.0050988, -0.0043848, 0.0053339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0025606
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0026927
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041610, -0.0037048, -0.0041423, -0.0036841, -0.0002539, 0.0002370
1: 0.0002614, 0.0027874, 0.0001469, 0.0026836, -0.0013120, 0.0014058
2: 0.0087387, 0.0143821, 0.0089707, 0.0146381, -0.0031408, 0.0029312
3: 0.0012737, 0.0036518, 0.0011658, 0.0035541, -0.0012352, 0.0013235
4: 1.0016918, 1.0109180, 1.0012733, 1.0105388, -0.0047921, 0.0051348
5: 0.0025705, 0.0043654, 0.0024891, 0.0042916, -0.0009323, 0.0009989
6: -0.0114239, -0.0090881, -0.0113278, -0.0089822, -0.0013000, 0.0012132
7: -0.0102606, -0.0099626, -0.0102483, -0.0099491, -0.0001658, 0.0001548
8: -0.0045749, -0.0029611, -0.0046481, -0.0030275, -0.0008382, 0.0008982
9: -0.0033468, 0.0047324, -0.0030147, 0.0050988, -0.0044965, 0.0041963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0025557
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0026921
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040916, -0.0036775, -0.0041424, -0.0036793, -0.0002534, 0.0003043
1: 0.0001100, 0.0024033, 0.0001204, 0.0026841, -0.0016848, 0.0014031
2: 0.0095969, 0.0147203, 0.0089695, 0.0146972, -0.0031348, 0.0037640
3: 0.0011312, 0.0032902, 0.0011409, 0.0035546, -0.0015862, 0.0013210
4: 1.0011388, 1.0095149, 1.0011766, 1.0105407, -0.0061537, 0.0051249
5: 0.0024629, 0.0040924, 0.0024703, 0.0042920, -0.0011971, 0.0009970
6: -0.0110687, -0.0089481, -0.0113284, -0.0089577, -0.0012975, 0.0015579
7: -0.0102153, -0.0099448, -0.0102484, -0.0099460, -0.0001655, 0.0001987
8: -0.0046717, -0.0032066, -0.0046651, -0.0030271, -0.0010764, 0.0008964
9: -0.0021182, 0.0052166, -0.0030164, 0.0051835, -0.0044878, 0.0053886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026069, upper bound: 0.0026908
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026069, upper bound: 0.0029130
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041497, -0.0036972, -0.0041424, -0.0036793, -0.0002609, 0.0002374
1: 0.0002196, 0.0027247, 0.0001204, 0.0026841, -0.0013144, 0.0014443
2: 0.0088788, 0.0144756, 0.0089695, 0.0146972, -0.0032268, 0.0029365
3: 0.0012343, 0.0035928, 0.0011409, 0.0035546, -0.0012374, 0.0013598
4: 1.0015389, 1.0106890, 1.0011766, 1.0105407, -0.0048008, 0.0052755
5: 0.0025408, 0.0043208, 0.0024703, 0.0042920, -0.0009340, 0.0010263
6: -0.0113659, -0.0090494, -0.0113284, -0.0089577, -0.0013356, 0.0012154
7: -0.0102532, -0.0099577, -0.0102484, -0.0099460, -0.0001704, 0.0001550
8: -0.0046017, -0.0030012, -0.0046651, -0.0030271, -0.0008397, 0.0009228
9: -0.0031463, 0.0048663, -0.0030164, 0.0051835, -0.0046196, 0.0042040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026069, upper bound: 0.0026908
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026069, upper bound: 0.0028594
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041090, -0.0036829, -0.0041482, -0.0036958, -0.0002535, 0.0002961
1: 0.0001400, 0.0024996, 0.0002116, 0.0027166, -0.0016393, 0.0014038
2: 0.0093817, 0.0146534, 0.0088969, 0.0144933, -0.0031362, 0.0036623
3: 0.0011594, 0.0033809, 0.0012268, 0.0035852, -0.0015433, 0.0013216
4: 1.0012481, 1.0098668, 1.0015100, 1.0106593, -0.0059874, 0.0051273
5: 0.0024842, 0.0041609, 0.0025351, 0.0043151, -0.0011648, 0.0009975
6: -0.0111578, -0.0089758, -0.0113584, -0.0090421, -0.0012981, 0.0015158
7: -0.0102266, -0.0099483, -0.0102522, -0.0099568, -0.0001656, 0.0001934
8: -0.0046525, -0.0031450, -0.0046068, -0.0030064, -0.0010473, 0.0008968
9: -0.0024263, 0.0051208, -0.0031203, 0.0048916, -0.0044898, 0.0052430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0025614
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0028379
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041641, -0.0037039, -0.0041482, -0.0036958, -0.0002671, 0.0002293
1: 0.0002563, 0.0028046, 0.0002116, 0.0027166, -0.0012696, 0.0014789
2: 0.0087004, 0.0143935, 0.0088969, 0.0144933, -0.0033040, 0.0028365
3: 0.0012689, 0.0036680, 0.0012268, 0.0035852, -0.0011953, 0.0013923
4: 1.0016732, 1.0109807, 1.0015100, 1.0106593, -0.0046373, 0.0054017
5: 0.0025669, 0.0043776, 0.0025351, 0.0043151, -0.0009021, 0.0010508
6: -0.0114397, -0.0090834, -0.0113584, -0.0090421, -0.0013675, 0.0011740
7: -0.0102626, -0.0099620, -0.0102522, -0.0099568, -0.0001744, 0.0001498
8: -0.0045782, -0.0029502, -0.0046068, -0.0030064, -0.0008111, 0.0009448
9: -0.0034017, 0.0047487, -0.0031203, 0.0048916, -0.0047301, 0.0040608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0025557
time: 0.70 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0027854
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040958, -0.0036767, -0.0041483, -0.0036907, -0.0002610, 0.0002999
1: 0.0001060, 0.0024263, 0.0001833, 0.0027172, -0.0016608, 0.0014449
2: 0.0095455, 0.0147292, 0.0088957, 0.0145566, -0.0032281, 0.0037104
3: 0.0011274, 0.0033118, 0.0012002, 0.0035857, -0.0015636, 0.0013603
4: 1.0011241, 1.0095990, 1.0014064, 1.0106614, -0.0060660, 0.0052776
5: 0.0024601, 0.0041088, 0.0025150, 0.0043155, -0.0011801, 0.0010267
6: -0.0110899, -0.0089444, -0.0113589, -0.0090159, -0.0013361, 0.0015357
7: -0.0102180, -0.0099443, -0.0102523, -0.0099534, -0.0001704, 0.0001959
8: -0.0046742, -0.0031919, -0.0046249, -0.0030060, -0.0010610, 0.0009231
9: -0.0021917, 0.0052294, -0.0031220, 0.0049822, -0.0046215, 0.0053119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028598
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0029636
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041533, -0.0036964, -0.0041483, -0.0036907, -0.0002753, 0.0002307
1: 0.0002147, 0.0027445, 0.0001833, 0.0027172, -0.0012775, 0.0015244
2: 0.0088346, 0.0144865, 0.0088957, 0.0145566, -0.0034058, 0.0028540
3: 0.0012297, 0.0036114, 0.0012002, 0.0035857, -0.0012027, 0.0014352
4: 1.0015212, 1.0107613, 1.0014064, 1.0106614, -0.0046660, 0.0055680
5: 0.0025373, 0.0043349, 0.0025150, 0.0043155, -0.0009077, 0.0010832
6: -0.0113842, -0.0090449, -0.0113589, -0.0090159, -0.0014096, 0.0011813
7: -0.0102555, -0.0099571, -0.0102523, -0.0099534, -0.0001798, 0.0001507
8: -0.0046048, -0.0029885, -0.0046249, -0.0030060, -0.0008162, 0.0009739
9: -0.0032095, 0.0048818, -0.0031220, 0.0049822, -0.0048758, 0.0040859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0027886
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028807
time: 0.69 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.43 seconds
IS_A1_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029556
IS_A1_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029557
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029836, upper bound: 0.0029164
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029836, upper bound: 0.0029429
IS_A1_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0030140
IS_A1_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0030235
IS_A1_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029429, upper bound: 0.0029966
IS_A1_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029429, upper bound: 0.0030014
IS_A1_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0028843
IS_A1_B1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0029559
IS_A1_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028418, upper bound: 0.0029695
IS_A1_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028418, upper bound: 0.0030350
IS_A1_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0028843
IS_A1_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028714, upper bound: 0.0030267
IS_A1_B1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028930, upper bound: 0.0029833
IS_A1_B1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028930, upper bound: 0.0030716
IS_A1_B1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0028513
IS_A1_B1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0028513
IS_A1_B1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0028513
IS_A1_B1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0028513
IS_A1_B1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029535, upper bound: 0.0028245
IS_A1_B1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029535, upper bound: 0.0028799
IS_A1_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029535, upper bound: 0.0028245
IS_A1_B1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029535, upper bound: 0.0028799
IS_A1_B1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0029225
IS_A1_B1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0029567
IS_A1_B1_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0029225
IS_A1_B1_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028800, upper bound: 0.0029567
IS_A1_B1_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029833, upper bound: 0.0029372
IS_A1_B1_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029833, upper bound: 0.0029503
IS_A1_B1_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029833, upper bound: 0.0029372
IS_A1_B1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029833, upper bound: 0.0029503
IS_A1_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0030950
IS_A1_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0031466
IS_A1_B2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028481, upper bound: 0.0031571
IS_A1_B2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028481, upper bound: 0.0032015
IS_A1_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0030950
IS_A1_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028824, upper bound: 0.0031826
IS_A1_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028996, upper bound: 0.0031802
IS_A1_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028996, upper bound: 0.0032137
IS_A1_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0027266
IS_A1_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0028134
IS_A1_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026186, upper bound: 0.0027528
IS_A1_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026186, upper bound: 0.0030565
IS_A1_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0027379
IS_A1_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027041, upper bound: 0.0029715
IS_A1_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028048, upper bound: 0.0030051
IS_A1_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028048, upper bound: 0.0031347
IS_A1_B2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0025799, upper bound: 0.0026477
IS_A1_B2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0025799, upper bound: 0.0026782
IS_A1_B2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0025799, upper bound: 0.0026477
IS_A1_B2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0025799, upper bound: 0.0026782
IS_A1_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027054, upper bound: 0.0026419
IS_A1_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027054, upper bound: 0.0029787
IS_A1_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027054, upper bound: 0.0026419
IS_A1_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027054, upper bound: 0.0029787
IS_A1_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0025801, upper bound: 0.0027371
IS_A1_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0025801, upper bound: 0.0029471
IS_A1_B2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0025801, upper bound: 0.0027371
IS_A1_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0025801, upper bound: 0.0029471
IS_A1_B2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028657, upper bound: 0.0030366
IS_A1_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028657, upper bound: 0.0030876
IS_A1_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028657, upper bound: 0.0030366
IS_A1_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0028657, upper bound: 0.0030876
IS_A2_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0028824
IS_A2_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0028824
IS_A2_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0031571, upper bound: 0.0028481
IS_A2_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0031571, upper bound: 0.0028996
IS_A2_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0029455
IS_A2_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0029657
IS_A2_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0031802, upper bound: 0.0029563
IS_A2_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0031802, upper bound: 0.0029608
IS_A2_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027266, upper bound: 0.0027041
IS_A2_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027266, upper bound: 0.0027041
IS_A2_B1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027528, upper bound: 0.0026186
IS_A2_B1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027528, upper bound: 0.0028048
IS_A2_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027379, upper bound: 0.0028021
IS_A2_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027379, upper bound: 0.0028416
IS_A2_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030051, upper bound: 0.0028783
IS_A2_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030051, upper bound: 0.0028955
IS_A2_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0028783
IS_A2_B1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0028783
IS_A2_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0031571, upper bound: 0.0028481
IS_A2_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0031571, upper bound: 0.0028996
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0029437
IS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030950, upper bound: 0.0029644
IS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0031802, upper bound: 0.0029562
IS_A2_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0031802, upper bound: 0.0029608
IS_A2_B1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027122, upper bound: 0.0026859
IS_A2_B1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027122, upper bound: 0.0026859
IS_A2_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027206, upper bound: 0.0026049
IS_A2_B1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027206, upper bound: 0.0028031
IS_A2_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027279, upper bound: 0.0027853
IS_A2_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0027279, upper bound: 0.0028346
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030051, upper bound: 0.0028782
IS_A2_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0030051, upper bound: 0.0028955
IS_A2_B2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0025799
IS_A2_B2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0027161
IS_A2_B2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0025718
IS_A2_B2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0027080
IS_A2_B2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026419, upper bound: 0.0027547
IS_A2_B2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026419, upper bound: 0.0029203
IS_A2_B2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026419, upper bound: 0.0027511
IS_A2_B2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026419, upper bound: 0.0028601
IS_A2_B2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0025801
IS_A2_B2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0028432
IS_A2_B2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0025718
IS_A2_B2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026477, upper bound: 0.0027877
IS_A2_B2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028657
IS_A2_B2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0029659
IS_A2_B2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0027911
IS_A2_B2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028811
IS_A2_B2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0025606
IS_A2_B2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0026927
IS_A2_B2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0025557
IS_A2_B2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0026921
IS_A2_B2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026069, upper bound: 0.0026908
IS_A2_B2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026069, upper bound: 0.0029130
IS_A2_B2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026069, upper bound: 0.0026908
IS_A2_B2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026069, upper bound: 0.0028594
IS_A2_B2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0025614
IS_A2_B2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0028379
IS_A2_B2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0025557
IS_A2_B2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0026033, upper bound: 0.0027854
IS_A2_B2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028598
IS_A2_B2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0029636
IS_A2_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0027886
IS_A2_B2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 4, lower bound: -0.0029787, upper bound: 0.0028807

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040001, -0.0036604, -0.0040132, -0.0036680, -0.0001990, 0.0002110
1: 0.0000155, 0.0018964, 0.0000577, 0.0019693, -0.0011683, 0.0011017
2: 0.0107294, 0.0149315, 0.0105665, 0.0148373, -0.0024614, 0.0026101
3: 0.0010422, 0.0028130, 0.0010819, 0.0028816, -0.0010999, 0.0010372
4: 1.0007935, 1.0076636, 1.0009476, 1.0079298, -0.0042672, 0.0040240
5: 0.0023958, 0.0037323, 0.0024257, 0.0037841, -0.0008301, 0.0007828
6: -0.0105999, -0.0088607, -0.0106674, -0.0088997, -0.0010187, 0.0010803
7: -0.0101555, -0.0099336, -0.0101641, -0.0099386, -0.0001300, 0.0001378
8: -0.0047321, -0.0035304, -0.0047051, -0.0034838, -0.0007464, 0.0007039
9: -0.0004969, 0.0055190, -0.0007301, 0.0053842, -0.0035237, 0.0037367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024613, upper bound: 0.0026508
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024227, upper bound: 0.0024347
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040001, -0.0036604, -0.0040247, -0.0036703, -0.0002028, 0.0002252
1: 0.0000155, 0.0018964, 0.0000707, 0.0020326, -0.0012469, 0.0011227
2: 0.0107294, 0.0149315, 0.0104252, 0.0148083, -0.0025082, 0.0027857
3: 0.0010422, 0.0028130, 0.0010941, 0.0029411, -0.0011739, 0.0010570
4: 1.0007935, 1.0076636, 1.0009949, 1.0081608, -0.0045542, 0.0041006
5: 0.0023958, 0.0037323, 0.0024350, 0.0038290, -0.0008860, 0.0007977
6: -0.0105999, -0.0088607, -0.0107258, -0.0089117, -0.0010381, 0.0011530
7: -0.0101555, -0.0099336, -0.0101715, -0.0099401, -0.0001324, 0.0001471
8: -0.0047321, -0.0035304, -0.0046968, -0.0034434, -0.0007966, 0.0007173
9: -0.0004969, 0.0055190, -0.0009324, 0.0053426, -0.0035908, 0.0039880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024613, upper bound: 0.0026508
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024227, upper bound: 0.0024675
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040132, -0.0036680, -0.0040097, -0.0036659, -0.0002201, 0.0002096
1: 0.0000577, 0.0019693, 0.0000459, 0.0019496, -0.0011605, 0.0012188
2: 0.0105665, 0.0148373, 0.0106105, 0.0148637, -0.0027228, 0.0025926
3: 0.0010819, 0.0028816, 0.0010708, 0.0028631, -0.0010925, 0.0011474
4: 1.0009476, 1.0079298, 1.0009044, 1.0078579, -0.0042387, 0.0044515
5: 0.0024257, 0.0037841, 0.0024173, 0.0037701, -0.0008246, 0.0008660
6: -0.0106674, -0.0088997, -0.0106491, -0.0088888, -0.0011270, 0.0010731
7: -0.0101641, -0.0099386, -0.0101618, -0.0099372, -0.0001438, 0.0001369
8: -0.0047051, -0.0034838, -0.0047127, -0.0034964, -0.0007414, 0.0007786
9: -0.0007301, 0.0053842, -0.0006671, 0.0054219, -0.0038981, 0.0037117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021628, upper bound: 0.0019382
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029164
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029164
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040001, -0.0036629, -0.0040097, -0.0036659, -0.0002018, 0.0002085
1: 0.0000296, 0.0018964, 0.0000459, 0.0019496, -0.0011546, 0.0011175
2: 0.0107294, 0.0149000, 0.0106105, 0.0148637, -0.0024965, 0.0025796
3: 0.0010554, 0.0028130, 0.0010708, 0.0028631, -0.0010870, 0.0010520
4: 1.0008451, 1.0076636, 1.0009044, 1.0078579, -0.0042173, 0.0040815
5: 0.0024058, 0.0037323, 0.0024173, 0.0037701, -0.0008204, 0.0007940
6: -0.0105999, -0.0088737, -0.0106491, -0.0088888, -0.0010333, 0.0010677
7: -0.0101555, -0.0099353, -0.0101618, -0.0099372, -0.0001318, 0.0001362
8: -0.0047231, -0.0035304, -0.0047127, -0.0034964, -0.0007377, 0.0007139
9: -0.0004969, 0.0054739, -0.0006671, 0.0054219, -0.0035741, 0.0036930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021628, upper bound: 0.0023198
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029428
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029429
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040117, -0.0036630, -0.0040132, -0.0036680, -0.0002153, 0.0002152
1: 0.0000302, 0.0019605, 0.0000577, 0.0019693, -0.0011914, 0.0011921
2: 0.0105862, 0.0148986, 0.0105665, 0.0148373, -0.0026632, 0.0026618
3: 0.0010560, 0.0028733, 0.0010819, 0.0028816, -0.0011217, 0.0011223
4: 1.0008473, 1.0078976, 1.0009476, 1.0079298, -0.0043517, 0.0043541
5: 0.0024062, 0.0037778, 0.0024257, 0.0037841, -0.0008466, 0.0008470
6: -0.0106592, -0.0088743, -0.0106674, -0.0088997, -0.0011023, 0.0011017
7: -0.0101630, -0.0099354, -0.0101641, -0.0099386, -0.0001406, 0.0001405
8: -0.0047227, -0.0034894, -0.0047051, -0.0034838, -0.0007612, 0.0007616
9: -0.0007019, 0.0054718, -0.0007301, 0.0053842, -0.0038128, 0.0038107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019382, upper bound: 0.0022421
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026218, upper bound: 0.0025414
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024277, upper bound: 0.0025234
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040117, -0.0036630, -0.0040247, -0.0036703, -0.0001982, 0.0002108
1: 0.0000302, 0.0019605, 0.0000707, 0.0020326, -0.0011673, 0.0010977
2: 0.0105862, 0.0148986, 0.0104252, 0.0148083, -0.0024524, 0.0026079
3: 0.0010560, 0.0028733, 0.0010941, 0.0029411, -0.0010990, 0.0010334
4: 1.0008473, 1.0078976, 1.0009949, 1.0081608, -0.0042636, 0.0040093
5: 0.0024062, 0.0037778, 0.0024350, 0.0038290, -0.0008294, 0.0007800
6: -0.0106592, -0.0088743, -0.0107258, -0.0089117, -0.0010150, 0.0010794
7: -0.0101630, -0.0099354, -0.0101715, -0.0099401, -0.0001295, 0.0001377
8: -0.0047227, -0.0034894, -0.0046968, -0.0034434, -0.0007458, 0.0007013
9: -0.0007019, 0.0054718, -0.0009324, 0.0053426, -0.0035109, 0.0037335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0019382, upper bound: 0.0027881
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024678, upper bound: 0.0028029
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024277, upper bound: 0.0027219
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040117, -0.0036590, -0.0040001, -0.0036629, -0.0002132, 0.0002218
1: 0.0000076, 0.0019610, 0.0000296, 0.0018964, -0.0012283, 0.0011807
2: 0.0105851, 0.0149491, 0.0107294, 0.0149000, -0.0026379, 0.0027442
3: 0.0010347, 0.0028738, 0.0010554, 0.0028130, -0.0011564, 0.0011116
4: 1.0007647, 1.0078994, 1.0008451, 1.0076636, -0.0044864, 0.0043126
5: 0.0023902, 0.0037781, 0.0024058, 0.0037323, -0.0008728, 0.0008390
6: -0.0106597, -0.0088534, -0.0105999, -0.0088737, -0.0010918, 0.0011358
7: -0.0101631, -0.0099327, -0.0101555, -0.0099353, -0.0001393, 0.0001449
8: -0.0047371, -0.0034891, -0.0047231, -0.0035304, -0.0007847, 0.0007543
9: -0.0007034, 0.0055442, -0.0004969, 0.0054739, -0.0037764, 0.0039286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029733
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029966
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040117, -0.0036590, -0.0040117, -0.0036657, -0.0001987, 0.0002161
1: 0.0000076, 0.0019610, 0.0000452, 0.0019605, -0.0011965, 0.0011004
2: 0.0105851, 0.0149491, 0.0105861, 0.0148652, -0.0024584, 0.0026731
3: 0.0010347, 0.0028738, 0.0010701, 0.0028733, -0.0011264, 0.0010360
4: 1.0007647, 1.0078994, 1.0009019, 1.0078977, -0.0043702, 0.0040193
5: 0.0023902, 0.0037781, 0.0024169, 0.0037778, -0.0008502, 0.0007819
6: -0.0106597, -0.0088534, -0.0106592, -0.0088882, -0.0010175, 0.0011064
7: -0.0101631, -0.0099327, -0.0101630, -0.0099371, -0.0001298, 0.0001411
8: -0.0047371, -0.0034891, -0.0047131, -0.0034894, -0.0007644, 0.0007030
9: -0.0007034, 0.0055442, -0.0007020, 0.0054240, -0.0035196, 0.0038268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0029827
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029164, upper bound: 0.0030014
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040132, -0.0036680, -0.0040587, -0.0036762, -0.0002171, 0.0002762
1: 0.0000577, 0.0019693, 0.0001028, 0.0022211, -0.0015290, 0.0012023
2: 0.0105665, 0.0148373, 0.0100039, 0.0147364, -0.0026860, 0.0034161
3: 0.0010819, 0.0028816, 0.0011244, 0.0031187, -0.0014395, 0.0011319
4: 1.0009476, 1.0079298, 1.0011126, 1.0088496, -0.0055848, 0.0043913
5: 0.0024257, 0.0037841, 0.0024578, 0.0039630, -0.0010865, 0.0008543
6: -0.0106674, -0.0088997, -0.0109002, -0.0089414, -0.0011117, 0.0014139
7: -0.0101641, -0.0099386, -0.0101938, -0.0099439, -0.0001418, 0.0001804
8: -0.0047051, -0.0034838, -0.0046763, -0.0033229, -0.0009769, 0.0007681
9: -0.0007301, 0.0053842, -0.0015355, 0.0052397, -0.0038454, 0.0048905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027301, upper bound: 0.0027145
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027341, upper bound: 0.0027079
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040247, -0.0036703, -0.0040587, -0.0036762, -0.0002313, 0.0002799
1: 0.0000707, 0.0020326, 0.0001028, 0.0022211, -0.0015500, 0.0012809
2: 0.0104252, 0.0148083, 0.0100039, 0.0147364, -0.0028616, 0.0034629
3: 0.0010941, 0.0029411, 0.0011244, 0.0031187, -0.0014593, 0.0012059
4: 1.0009949, 1.0081608, 1.0011126, 1.0088496, -0.0056614, 0.0046784
5: 0.0024350, 0.0038290, 0.0024578, 0.0039630, -0.0011014, 0.0009101
6: -0.0107258, -0.0089117, -0.0109002, -0.0089414, -0.0011844, 0.0014333
7: -0.0101715, -0.0099401, -0.0101938, -0.0099439, -0.0001511, 0.0001828
8: -0.0046968, -0.0034434, -0.0046763, -0.0033229, -0.0009903, 0.0008183
9: -0.0009324, 0.0053426, -0.0015355, 0.0052397, -0.0040967, 0.0049575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.38 + 598.08 = 601.46 seconds
