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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041062, -0.0040755, -0.0041062, -0.0040755, -0.0000166, 0.0000166)
1: (-0.0064338, -0.0052844, -0.0064338, -0.0052844, -0.0006222, 0.0006222)
2: (0.9687427, 0.9701220, 0.9687427, 0.9701220, -0.0007467, 0.0007467)
3: (0.0157570, 0.0259306, 0.0157570, 0.0259306, -0.0055073, 0.0055073)
4: (-0.0026652, -0.0018914, -0.0026652, -0.0018914, -0.0004189, 0.0004189)
5: (0.0145767, 0.0153587, 0.0145767, 0.0153587, -0.0004233, 0.0004233)
6: (0.0044363, 0.0048167, 0.0044363, 0.0048167, -0.0002059, 0.0002059)
7: (-0.0144984, -0.0118618, -0.0144984, -0.0118618, -0.0014273, 0.0014273)
8: (0.0052268, 0.0073185, 0.0052268, 0.0073185, -0.0011323, 0.0011323)
9: (0.0071255, 0.0108877, 0.0071255, 0.0108877, -0.0020366, 0.0020366)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.64 = 2.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0005113, upper bound: 0.0005113

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004917, upper bound: 0.0004938
time: 0.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004938, upper bound: 0.0004938
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 2, lower bound: -0.0004917, upper bound: 0.0004938
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 2, lower bound: -0.0004938, upper bound: 0.0004938

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040756, -0.0041060, -0.0040755, -0.0000158, 0.0000163
1: -0.0064059, -0.0052858, -0.0064272, -0.0052847, -0.0005930, 0.0006094
2: 0.9687761, 0.9701203, 0.9687507, 0.9701216, -0.0007116, 0.0007313
3: 0.0160041, 0.0259183, 0.0158156, 0.0259275, -0.0052484, 0.0053939
4: -0.0026643, -0.0019102, -0.0026650, -0.0018959, -0.0004102, 0.0003992
5: 0.0145776, 0.0153397, 0.0145769, 0.0153542, -0.0004146, 0.0004034
6: 0.0044456, 0.0048162, 0.0044385, 0.0048166, -0.0001962, 0.0002017
7: -0.0144952, -0.0119259, -0.0144976, -0.0118770, -0.0013979, 0.0013602
8: 0.0052293, 0.0072677, 0.0052274, 0.0073065, -0.0011090, 0.0010791
9: 0.0071301, 0.0107963, 0.0071267, 0.0108661, -0.0019946, 0.0019409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0004675
time: 0.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004804, upper bound: 0.0004828
time: 0.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040751, -0.0041060, -0.0040755, -0.0000159, 0.0000167
1: -0.0064055, -0.0052687, -0.0064242, -0.0052850, -0.0005966, 0.0006254
2: 0.9687765, 0.9701408, 0.9687542, 0.9701213, -0.0007159, 0.0007506
3: 0.0160072, 0.0260696, 0.0158416, 0.0259256, -0.0052805, 0.0055360
4: -0.0026758, -0.0019105, -0.0026648, -0.0018979, -0.0004210, 0.0004016
5: 0.0145660, 0.0153395, 0.0145771, 0.0153522, -0.0004255, 0.0004059
6: 0.0044457, 0.0048219, 0.0044395, 0.0048165, -0.0001974, 0.0002070
7: -0.0145344, -0.0119267, -0.0144971, -0.0118837, -0.0014347, 0.0013685
8: 0.0051982, 0.0072671, 0.0052278, 0.0073011, -0.0011382, 0.0010857
9: 0.0070741, 0.0107952, 0.0071274, 0.0108564, -0.0020472, 0.0019527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004827, upper bound: 0.0004675
time: 0.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004828
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.78 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0004675
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 2, lower bound: -0.0004804, upper bound: 0.0004828
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 2, lower bound: -0.0004827, upper bound: 0.0004675
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004828

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040756, -0.0041051, -0.0040759, -0.0000146, 0.0000152
1: -0.0063951, -0.0052869, -0.0063906, -0.0052971, -0.0005482, 0.0005684
2: 0.9687891, 0.9701189, 0.9687945, 0.9701068, -0.0006578, 0.0006822
3: 0.0160998, 0.0259084, 0.0161393, 0.0258185, -0.0048519, 0.0050315
4: -0.0026635, -0.0019175, -0.0026567, -0.0019205, -0.0003827, 0.0003690
5: 0.0145784, 0.0153324, 0.0145853, 0.0153293, -0.0003868, 0.0003730
6: 0.0044491, 0.0048159, 0.0044506, 0.0048125, -0.0001814, 0.0001881
7: -0.0144926, -0.0119507, -0.0144694, -0.0119609, -0.0013040, 0.0012574
8: 0.0052314, 0.0072481, 0.0052498, 0.0072399, -0.0010345, 0.0009976
9: 0.0071337, 0.0107609, 0.0071670, 0.0107463, -0.0018606, 0.0017942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0004667
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0004675
time: 0.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040756, -0.0041056, -0.0040756, -0.0000157, 0.0000147
1: -0.0064014, -0.0052862, -0.0064095, -0.0052865, -0.0005879, 0.0005520
2: 0.9687814, 0.9701198, 0.9687718, 0.9701194, -0.0007055, 0.0006624
3: 0.0160433, 0.0259145, 0.0159716, 0.0259115, -0.0052036, 0.0048856
4: -0.0026640, -0.0019132, -0.0026638, -0.0019078, -0.0003716, 0.0003958
5: 0.0145779, 0.0153367, 0.0145782, 0.0153422, -0.0003755, 0.0004000
6: 0.0044470, 0.0048161, 0.0044443, 0.0048160, -0.0001946, 0.0001827
7: -0.0144942, -0.0119360, -0.0144935, -0.0119174, -0.0012661, 0.0013485
8: 0.0052301, 0.0072597, 0.0052307, 0.0072744, -0.0010045, 0.0010699
9: 0.0071315, 0.0107818, 0.0071326, 0.0108084, -0.0018067, 0.0019243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004804, upper bound: 0.0004804
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004804, upper bound: 0.0004828
time: 0.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041050, -0.0040759, -0.0000147, 0.0000156
1: -0.0063944, -0.0052698, -0.0063875, -0.0052973, -0.0005518, 0.0005841
2: 0.9687899, 0.9701395, 0.9687982, 0.9701065, -0.0006621, 0.0007009
3: 0.0161056, 0.0260596, 0.0161667, 0.0258168, -0.0048838, 0.0051698
4: -0.0026750, -0.0019180, -0.0026565, -0.0019226, -0.0003932, 0.0003714
5: 0.0145668, 0.0153319, 0.0145854, 0.0153272, -0.0003974, 0.0003754
6: 0.0044493, 0.0048215, 0.0044516, 0.0048124, -0.0001826, 0.0001933
7: -0.0145318, -0.0119522, -0.0144689, -0.0119680, -0.0013398, 0.0012657
8: 0.0052003, 0.0072469, 0.0052502, 0.0072343, -0.0010629, 0.0010041
9: 0.0070778, 0.0107588, 0.0071676, 0.0107362, -0.0019118, 0.0018060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004667
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004667
time: 0.79 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040751, -0.0041055, -0.0040756, -0.0000158, 0.0000153
1: -0.0064013, -0.0052691, -0.0064069, -0.0052868, -0.0005915, 0.0005733
2: 0.9687818, 0.9701403, 0.9687749, 0.9701191, -0.0007099, 0.0006880
3: 0.0160449, 0.0260658, 0.0159947, 0.0259097, -0.0052360, 0.0050747
4: -0.0026755, -0.0019133, -0.0026636, -0.0019095, -0.0003860, 0.0003982
5: 0.0145663, 0.0153366, 0.0145783, 0.0153404, -0.0003901, 0.0004025
6: 0.0044471, 0.0048217, 0.0044452, 0.0048159, -0.0001958, 0.0001897
7: -0.0145334, -0.0119364, -0.0144930, -0.0119234, -0.0013151, 0.0013569
8: 0.0051990, 0.0072594, 0.0052311, 0.0072697, -0.0010434, 0.0010765
9: 0.0070756, 0.0107812, 0.0071332, 0.0107998, -0.0018766, 0.0019363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004804
time: 0.76 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004804
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.83 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0004667
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0004675
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 2, lower bound: -0.0004804, upper bound: 0.0004804
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 2, lower bound: -0.0004804, upper bound: 0.0004828
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004667
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004667
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004804
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 2, lower bound: -0.0004828, upper bound: 0.0004804

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040756, -0.0041045, -0.0040759, -0.0000145, 0.0000146
1: -0.0063951, -0.0052869, -0.0063697, -0.0052979, -0.0005441, 0.0005471
2: 0.9687891, 0.9701189, 0.9688196, 0.9701058, -0.0006530, 0.0006566
3: 0.0160998, 0.0259084, 0.0163246, 0.0258114, -0.0048160, 0.0048429
4: -0.0026635, -0.0019175, -0.0026561, -0.0019346, -0.0003683, 0.0003663
5: 0.0145784, 0.0153324, 0.0145858, 0.0153151, -0.0003723, 0.0003702
6: 0.0044491, 0.0048159, 0.0044575, 0.0048122, -0.0001801, 0.0001811
7: -0.0144926, -0.0119507, -0.0144675, -0.0120089, -0.0012551, 0.0012481
8: 0.0052314, 0.0072481, 0.0052513, 0.0072018, -0.0009957, 0.0009902
9: 0.0071337, 0.0107609, 0.0071696, 0.0106778, -0.0017909, 0.0017810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004715, upper bound: 0.0004667
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004715, upper bound: 0.0004667
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040756, -0.0041045, -0.0040755, -0.0000151, 0.0000147
1: -0.0063951, -0.0052869, -0.0063682, -0.0052820, -0.0005667, 0.0005506
2: 0.9687891, 0.9701189, 0.9688213, 0.9701248, -0.0006801, 0.0006607
3: 0.0160998, 0.0259084, 0.0163371, 0.0259519, -0.0050160, 0.0048731
4: -0.0026635, -0.0019175, -0.0026668, -0.0019356, -0.0003706, 0.0003815
5: 0.0145784, 0.0153324, 0.0145751, 0.0153141, -0.0003746, 0.0003856
6: 0.0044491, 0.0048159, 0.0044580, 0.0048175, -0.0001875, 0.0001822
7: -0.0144926, -0.0119507, -0.0145039, -0.0120122, -0.0012629, 0.0012999
8: 0.0052314, 0.0072481, 0.0052224, 0.0071993, -0.0010019, 0.0010313
9: 0.0071337, 0.0107609, 0.0071177, 0.0106732, -0.0018021, 0.0018549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004715, upper bound: 0.0004675
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004715, upper bound: 0.0004675
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040756, -0.0041050, -0.0040756, -0.0000156, 0.0000142
1: -0.0064014, -0.0052862, -0.0063881, -0.0052875, -0.0005837, 0.0005308
2: 0.9687814, 0.9701198, 0.9687974, 0.9701183, -0.0007005, 0.0006369
3: 0.0160433, 0.0259145, 0.0161617, 0.0259031, -0.0051666, 0.0046979
4: -0.0026640, -0.0019132, -0.0026631, -0.0019222, -0.0003573, 0.0003930
5: 0.0145779, 0.0153367, 0.0145788, 0.0153276, -0.0003611, 0.0003971
6: 0.0044470, 0.0048161, 0.0044514, 0.0048157, -0.0001932, 0.0001756
7: -0.0144942, -0.0119360, -0.0144913, -0.0119667, -0.0012175, 0.0013390
8: 0.0052301, 0.0072597, 0.0052325, 0.0072353, -0.0009659, 0.0010623
9: 0.0071315, 0.0107818, 0.0071357, 0.0107380, -0.0017373, 0.0019106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004671, upper bound: 0.0004803
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004671, upper bound: 0.0004669
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040756, -0.0041050, -0.0040752, -0.0000161, 0.0000143
1: -0.0064014, -0.0052862, -0.0063883, -0.0052704, -0.0006011, 0.0005341
2: 0.9687814, 0.9701198, 0.9687972, 0.9701387, -0.0007214, 0.0006409
3: 0.0160433, 0.0259145, 0.0161593, 0.0260542, -0.0053206, 0.0047271
4: -0.0026640, -0.0019132, -0.0026746, -0.0019220, -0.0003595, 0.0004047
5: 0.0145779, 0.0153367, 0.0145672, 0.0153278, -0.0003634, 0.0004090
6: 0.0044470, 0.0048161, 0.0044514, 0.0048213, -0.0001989, 0.0001767
7: -0.0144942, -0.0119360, -0.0145304, -0.0119661, -0.0012251, 0.0013789
8: 0.0052301, 0.0072597, 0.0052014, 0.0072358, -0.0009719, 0.0010939
9: 0.0071315, 0.0107818, 0.0070798, 0.0107389, -0.0017481, 0.0019676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004671, upper bound: 0.0004827
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004671, upper bound: 0.0004677
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041045, -0.0040759, -0.0000146, 0.0000151
1: -0.0063944, -0.0052698, -0.0063697, -0.0052979, -0.0005472, 0.0005646
2: 0.9687899, 0.9701395, 0.9688196, 0.9701058, -0.0006567, 0.0006776
3: 0.0161056, 0.0260596, 0.0163246, 0.0258114, -0.0048435, 0.0049979
4: -0.0026750, -0.0019180, -0.0026561, -0.0019346, -0.0003801, 0.0003684
5: 0.0145668, 0.0153319, 0.0145858, 0.0153151, -0.0003842, 0.0003723
6: 0.0044493, 0.0048215, 0.0044575, 0.0048122, -0.0001811, 0.0001869
7: -0.0145318, -0.0119522, -0.0144675, -0.0120089, -0.0012952, 0.0012552
8: 0.0052003, 0.0072469, 0.0052513, 0.0072018, -0.0010276, 0.0009958
9: 0.0070778, 0.0107588, 0.0071696, 0.0106778, -0.0018482, 0.0017911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004667
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004667
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041045, -0.0040755, -0.0000147, 0.0000147
1: -0.0063944, -0.0052698, -0.0063682, -0.0052820, -0.0005492, 0.0005518
2: 0.9687899, 0.9701395, 0.9688213, 0.9701248, -0.0006591, 0.0006622
3: 0.0161056, 0.0260596, 0.0163371, 0.0259519, -0.0048613, 0.0048841
4: -0.0026750, -0.0019180, -0.0026668, -0.0019356, -0.0003715, 0.0003697
5: 0.0145668, 0.0153319, 0.0145751, 0.0153141, -0.0003754, 0.0003737
6: 0.0044493, 0.0048215, 0.0044580, 0.0048175, -0.0001818, 0.0001826
7: -0.0145318, -0.0119522, -0.0145039, -0.0120122, -0.0012658, 0.0012598
8: 0.0052003, 0.0072469, 0.0052224, 0.0071993, -0.0010042, 0.0009995
9: 0.0070778, 0.0107588, 0.0071177, 0.0106732, -0.0018061, 0.0017977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004667
time: 1.01 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004667
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040751, -0.0041050, -0.0040756, -0.0000156, 0.0000148
1: -0.0064013, -0.0052691, -0.0063881, -0.0052875, -0.0005860, 0.0005540
2: 0.9687818, 0.9701403, 0.9687974, 0.9701183, -0.0007032, 0.0006649
3: 0.0160449, 0.0260658, 0.0161617, 0.0259031, -0.0051870, 0.0049039
4: -0.0026755, -0.0019133, -0.0026631, -0.0019222, -0.0003730, 0.0003945
5: 0.0145663, 0.0153366, 0.0145788, 0.0153276, -0.0003769, 0.0003987
6: 0.0044471, 0.0048217, 0.0044514, 0.0048157, -0.0001939, 0.0001833
7: -0.0145334, -0.0119364, -0.0144913, -0.0119667, -0.0012709, 0.0013443
8: 0.0051990, 0.0072594, 0.0052325, 0.0072353, -0.0010083, 0.0010665
9: 0.0070756, 0.0107812, 0.0071357, 0.0107380, -0.0018134, 0.0019182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004675, upper bound: 0.0004803
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004675, upper bound: 0.0004669
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040751, -0.0041050, -0.0040752, -0.0000157, 0.0000143
1: -0.0064013, -0.0052691, -0.0063883, -0.0052704, -0.0005887, 0.0005357
2: 0.9687818, 0.9701403, 0.9687972, 0.9701387, -0.0007065, 0.0006428
3: 0.0160449, 0.0260658, 0.0161593, 0.0260542, -0.0052108, 0.0047414
4: -0.0026755, -0.0019133, -0.0026746, -0.0019220, -0.0003606, 0.0003963
5: 0.0145663, 0.0153366, 0.0145672, 0.0153278, -0.0003645, 0.0004005
6: 0.0044471, 0.0048217, 0.0044514, 0.0048213, -0.0001948, 0.0001773
7: -0.0145334, -0.0119364, -0.0145304, -0.0119661, -0.0012288, 0.0013504
8: 0.0051990, 0.0072594, 0.0052014, 0.0072358, -0.0009749, 0.0010714
9: 0.0070756, 0.0107812, 0.0070798, 0.0107389, -0.0017534, 0.0019269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004675, upper bound: 0.0004803
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004675, upper bound: 0.0004669
time: 1.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.16 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004715, upper bound: 0.0004667
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004715, upper bound: 0.0004667
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004715, upper bound: 0.0004675
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004715, upper bound: 0.0004675
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004671, upper bound: 0.0004803
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004671, upper bound: 0.0004669
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004671, upper bound: 0.0004827
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004671, upper bound: 0.0004677
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004667
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004667
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004667
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004667
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004675, upper bound: 0.0004803
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004675, upper bound: 0.0004669
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004675, upper bound: 0.0004803
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0004675, upper bound: 0.0004669

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040759, -0.0041045, -0.0040759, -0.0000138, 0.0000138
1: -0.0063697, -0.0052979, -0.0063697, -0.0052979, -0.0005166, 0.0005166
2: 0.9688196, 0.9701058, 0.9688196, 0.9701058, -0.0006199, 0.0006199
3: 0.0163246, 0.0258114, 0.0163246, 0.0258114, -0.0045722, 0.0045722
4: -0.0026561, -0.0019346, -0.0026561, -0.0019346, -0.0003477, 0.0003477
5: 0.0145858, 0.0153151, 0.0145858, 0.0153151, -0.0003515, 0.0003515
6: 0.0044575, 0.0048122, 0.0044575, 0.0048122, -0.0001709, 0.0001709
7: -0.0144675, -0.0120089, -0.0144675, -0.0120089, -0.0011849, 0.0011849
8: 0.0052513, 0.0072018, 0.0052513, 0.0072018, -0.0009401, 0.0009401
9: 0.0071696, 0.0106778, 0.0071696, 0.0106778, -0.0016908, 0.0016908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004515
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004616, upper bound: 0.0004572
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040756, -0.0041045, -0.0040759, -0.0000145, 0.0000146
1: -0.0063881, -0.0052875, -0.0063697, -0.0052979, -0.0005436, 0.0005467
2: 0.9687974, 0.9701183, 0.9688196, 0.9701058, -0.0006524, 0.0006561
3: 0.0161617, 0.0259031, 0.0163246, 0.0258114, -0.0048118, 0.0048392
4: -0.0026631, -0.0019222, -0.0026561, -0.0019346, -0.0003681, 0.0003660
5: 0.0145788, 0.0153276, 0.0145858, 0.0153151, -0.0003720, 0.0003699
6: 0.0044514, 0.0048157, 0.0044575, 0.0048122, -0.0001799, 0.0001809
7: -0.0144913, -0.0119667, -0.0144675, -0.0120089, -0.0012541, 0.0012470
8: 0.0052325, 0.0072353, 0.0052513, 0.0072018, -0.0009950, 0.0009893
9: 0.0071357, 0.0107380, 0.0071696, 0.0106778, -0.0017895, 0.0017794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004556, upper bound: 0.0004506
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004616, upper bound: 0.0004572
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040759, -0.0041045, -0.0040755, -0.0000144, 0.0000139
1: -0.0063697, -0.0052979, -0.0063682, -0.0052820, -0.0005391, 0.0005200
2: 0.9688196, 0.9701058, 0.9688213, 0.9701248, -0.0006470, 0.0006240
3: 0.0163246, 0.0258114, 0.0163371, 0.0259519, -0.0047721, 0.0046023
4: -0.0026561, -0.0019346, -0.0026668, -0.0019356, -0.0003500, 0.0003629
5: 0.0145858, 0.0153151, 0.0145751, 0.0153141, -0.0003538, 0.0003668
6: 0.0044575, 0.0048122, 0.0044580, 0.0048175, -0.0001784, 0.0001721
7: -0.0144675, -0.0120089, -0.0145039, -0.0120122, -0.0011927, 0.0012367
8: 0.0052513, 0.0072018, 0.0052224, 0.0071993, -0.0009463, 0.0009812
9: 0.0071696, 0.0106778, 0.0071177, 0.0106732, -0.0017019, 0.0017647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004534, upper bound: 0.0004509
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004573
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040756, -0.0041045, -0.0040755, -0.0000151, 0.0000147
1: -0.0063881, -0.0052875, -0.0063682, -0.0052820, -0.0005662, 0.0005501
2: 0.9687974, 0.9701183, 0.9688213, 0.9701248, -0.0006795, 0.0006602
3: 0.0161617, 0.0259031, 0.0163371, 0.0259519, -0.0050118, 0.0048694
4: -0.0026631, -0.0019222, -0.0026668, -0.0019356, -0.0003703, 0.0003812
5: 0.0145788, 0.0153276, 0.0145751, 0.0153141, -0.0003743, 0.0003852
6: 0.0044514, 0.0048157, 0.0044580, 0.0048175, -0.0001874, 0.0001821
7: -0.0144913, -0.0119667, -0.0145039, -0.0120122, -0.0012620, 0.0012989
8: 0.0052325, 0.0072353, 0.0052224, 0.0071993, -0.0010012, 0.0010304
9: 0.0071357, 0.0107380, 0.0071177, 0.0106732, -0.0018007, 0.0018534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004534, upper bound: 0.0004509
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004573
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040759, -0.0041050, -0.0040756, -0.0000146, 0.0000145
1: -0.0063697, -0.0052979, -0.0063881, -0.0052875, -0.0005467, 0.0005436
2: 0.9688196, 0.9701058, 0.9687974, 0.9701183, -0.0006561, 0.0006524
3: 0.0163246, 0.0258114, 0.0161617, 0.0259031, -0.0048392, 0.0048118
4: -0.0026561, -0.0019346, -0.0026631, -0.0019222, -0.0003660, 0.0003681
5: 0.0145858, 0.0153151, 0.0145788, 0.0153276, -0.0003699, 0.0003720
6: 0.0044575, 0.0048122, 0.0044514, 0.0048157, -0.0001809, 0.0001799
7: -0.0144675, -0.0120089, -0.0144913, -0.0119667, -0.0012470, 0.0012541
8: 0.0052513, 0.0072018, 0.0052325, 0.0072353, -0.0009893, 0.0009950
9: 0.0071696, 0.0106778, 0.0071357, 0.0107380, -0.0017794, 0.0017895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004646
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004572, upper bound: 0.0004705
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040756, -0.0041050, -0.0040756, -0.0000141, 0.0000141
1: -0.0063881, -0.0052875, -0.0063881, -0.0052875, -0.0005298, 0.0005298
2: 0.9687974, 0.9701183, 0.9687974, 0.9701183, -0.0006358, 0.0006358
3: 0.0161617, 0.0259031, 0.0161617, 0.0259031, -0.0046897, 0.0046897
4: -0.0026631, -0.0019222, -0.0026631, -0.0019222, -0.0003567, 0.0003567
5: 0.0145788, 0.0153276, 0.0145788, 0.0153276, -0.0003605, 0.0003605
6: 0.0044514, 0.0048157, 0.0044514, 0.0048157, -0.0001753, 0.0001753
7: -0.0144913, -0.0119667, -0.0144913, -0.0119667, -0.0012154, 0.0012154
8: 0.0052325, 0.0072353, 0.0052325, 0.0072353, -0.0009642, 0.0009642
9: 0.0071357, 0.0107380, 0.0071357, 0.0107380, -0.0017342, 0.0017342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004522
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004572, upper bound: 0.0004575
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040759, -0.0041050, -0.0040752, -0.0000151, 0.0000145
1: -0.0063697, -0.0052979, -0.0063883, -0.0052704, -0.0005641, 0.0005448
2: 0.9688196, 0.9701058, 0.9687972, 0.9701387, -0.0006770, 0.0006538
3: 0.0163246, 0.0258114, 0.0161593, 0.0260542, -0.0049932, 0.0048226
4: -0.0026561, -0.0019346, -0.0026746, -0.0019220, -0.0003668, 0.0003798
5: 0.0145858, 0.0153151, 0.0145672, 0.0153278, -0.0003707, 0.0003838
6: 0.0044575, 0.0048122, 0.0044514, 0.0048213, -0.0001867, 0.0001803
7: -0.0144675, -0.0120089, -0.0145304, -0.0119661, -0.0012498, 0.0012940
8: 0.0052513, 0.0072018, 0.0052014, 0.0072358, -0.0009915, 0.0010266
9: 0.0071696, 0.0106778, 0.0070798, 0.0107389, -0.0017834, 0.0018465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004659
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004569, upper bound: 0.0004726
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040756, -0.0041050, -0.0040752, -0.0000148, 0.0000142
1: -0.0063881, -0.0052875, -0.0063883, -0.0052704, -0.0005528, 0.0005331
2: 0.9687974, 0.9701183, 0.9687972, 0.9701387, -0.0006633, 0.0006398
3: 0.0161617, 0.0259031, 0.0161593, 0.0260542, -0.0048926, 0.0047189
4: -0.0026631, -0.0019222, -0.0026746, -0.0019220, -0.0003589, 0.0003721
5: 0.0145788, 0.0153276, 0.0145672, 0.0153278, -0.0003627, 0.0003761
6: 0.0044514, 0.0048157, 0.0044514, 0.0048213, -0.0001829, 0.0001764
7: -0.0144913, -0.0119667, -0.0145304, -0.0119661, -0.0012229, 0.0012680
8: 0.0052325, 0.0072353, 0.0052014, 0.0072358, -0.0009702, 0.0010059
9: 0.0071357, 0.0107380, 0.0070798, 0.0107389, -0.0017450, 0.0018093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004516
time: 0.99 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004569, upper bound: 0.0004577
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040755, -0.0041045, -0.0040759, -0.0000139, 0.0000144
1: -0.0063682, -0.0052820, -0.0063697, -0.0052979, -0.0005200, 0.0005391
2: 0.9688213, 0.9701248, 0.9688196, 0.9701058, -0.0006240, 0.0006470
3: 0.0163371, 0.0259519, 0.0163246, 0.0258114, -0.0046023, 0.0047721
4: -0.0026668, -0.0019356, -0.0026561, -0.0019346, -0.0003629, 0.0003500
5: 0.0145751, 0.0153141, 0.0145858, 0.0153151, -0.0003668, 0.0003538
6: 0.0044580, 0.0048175, 0.0044575, 0.0048122, -0.0001721, 0.0001784
7: -0.0145039, -0.0120122, -0.0144675, -0.0120089, -0.0012367, 0.0011927
8: 0.0052224, 0.0071993, 0.0052513, 0.0072018, -0.0009812, 0.0009463
9: 0.0071177, 0.0106732, 0.0071696, 0.0106778, -0.0017647, 0.0017019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004499
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004569
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040752, -0.0041045, -0.0040759, -0.0000145, 0.0000151
1: -0.0063883, -0.0052704, -0.0063697, -0.0052979, -0.0005448, 0.0005641
2: 0.9687972, 0.9701387, 0.9688196, 0.9701058, -0.0006538, 0.0006770
3: 0.0161593, 0.0260542, 0.0163246, 0.0258114, -0.0048226, 0.0049932
4: -0.0026746, -0.0019220, -0.0026561, -0.0019346, -0.0003798, 0.0003668
5: 0.0145672, 0.0153278, 0.0145858, 0.0153151, -0.0003838, 0.0003707
6: 0.0044514, 0.0048213, 0.0044575, 0.0048122, -0.0001803, 0.0001867
7: -0.0145304, -0.0119661, -0.0144675, -0.0120089, -0.0012940, 0.0012498
8: 0.0052014, 0.0072358, 0.0052513, 0.0072018, -0.0010266, 0.0009915
9: 0.0070798, 0.0107389, 0.0071696, 0.0106778, -0.0018465, 0.0017834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004499
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004569
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040755, -0.0041045, -0.0040755, -0.0000139, 0.0000139
1: -0.0063682, -0.0052820, -0.0063682, -0.0052820, -0.0005214, 0.0005214
2: 0.9688213, 0.9701248, 0.9688213, 0.9701248, -0.0006257, 0.0006257
3: 0.0163371, 0.0259519, 0.0163371, 0.0259519, -0.0046153, 0.0046153
4: -0.0026668, -0.0019356, -0.0026668, -0.0019356, -0.0003510, 0.0003510
5: 0.0145751, 0.0153141, 0.0145751, 0.0153141, -0.0003548, 0.0003548
6: 0.0044580, 0.0048175, 0.0044580, 0.0048175, -0.0001726, 0.0001726
7: -0.0145039, -0.0120122, -0.0145039, -0.0120122, -0.0011961, 0.0011961
8: 0.0052224, 0.0071993, 0.0052224, 0.0071993, -0.0009489, 0.0009489
9: 0.0071177, 0.0106732, 0.0071177, 0.0106732, -0.0017067, 0.0017067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004502
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004565
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040752, -0.0041045, -0.0040755, -0.0000147, 0.0000147
1: -0.0063883, -0.0052704, -0.0063682, -0.0052820, -0.0005487, 0.0005514
2: 0.9687972, 0.9701387, 0.9688213, 0.9701248, -0.0006584, 0.0006617
3: 0.0161593, 0.0260542, 0.0163371, 0.0259519, -0.0048563, 0.0048806
4: -0.0026746, -0.0019220, -0.0026668, -0.0019356, -0.0003712, 0.0003694
5: 0.0145672, 0.0153278, 0.0145751, 0.0153141, -0.0003752, 0.0003733
6: 0.0044514, 0.0048213, 0.0044580, 0.0048175, -0.0001816, 0.0001825
7: -0.0145304, -0.0119661, -0.0145039, -0.0120122, -0.0012648, 0.0012586
8: 0.0052014, 0.0072358, 0.0052224, 0.0071993, -0.0010035, 0.0009985
9: 0.0070798, 0.0107389, 0.0071177, 0.0106732, -0.0018048, 0.0017959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004493
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004565
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040755, -0.0041050, -0.0040756, -0.0000147, 0.0000151
1: -0.0063682, -0.0052820, -0.0063881, -0.0052875, -0.0005501, 0.0005662
2: 0.9688213, 0.9701248, 0.9687974, 0.9701183, -0.0006602, 0.0006795
3: 0.0163371, 0.0259519, 0.0161617, 0.0259031, -0.0048694, 0.0050118
4: -0.0026668, -0.0019356, -0.0026631, -0.0019222, -0.0003812, 0.0003703
5: 0.0145751, 0.0153141, 0.0145788, 0.0153276, -0.0003852, 0.0003743
6: 0.0044580, 0.0048175, 0.0044514, 0.0048157, -0.0001821, 0.0001874
7: -0.0145039, -0.0120122, -0.0144913, -0.0119667, -0.0012989, 0.0012620
8: 0.0052224, 0.0071993, 0.0052325, 0.0072353, -0.0010304, 0.0010012
9: 0.0071177, 0.0106732, 0.0071357, 0.0107380, -0.0018534, 0.0018007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004509, upper bound: 0.0004625
time: 0.81 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004573, upper bound: 0.0004702
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040752, -0.0041050, -0.0040756, -0.0000142, 0.0000148
1: -0.0063883, -0.0052704, -0.0063881, -0.0052875, -0.0005331, 0.0005528
2: 0.9687972, 0.9701387, 0.9687974, 0.9701183, -0.0006398, 0.0006633
3: 0.0161593, 0.0260542, 0.0161617, 0.0259031, -0.0047189, 0.0048926
4: -0.0026746, -0.0019220, -0.0026631, -0.0019222, -0.0003721, 0.0003589
5: 0.0145672, 0.0153278, 0.0145788, 0.0153276, -0.0003761, 0.0003627
6: 0.0044514, 0.0048213, 0.0044514, 0.0048157, -0.0001764, 0.0001829
7: -0.0145304, -0.0119661, -0.0144913, -0.0119667, -0.0012680, 0.0012229
8: 0.0052014, 0.0072358, 0.0052325, 0.0072353, -0.0010059, 0.0009702
9: 0.0070798, 0.0107389, 0.0071357, 0.0107380, -0.0018093, 0.0017450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004509, upper bound: 0.0004505
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004573, upper bound: 0.0004572
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040755, -0.0041050, -0.0040752, -0.0000147, 0.0000147
1: -0.0063682, -0.0052820, -0.0063883, -0.0052704, -0.0005514, 0.0005487
2: 0.9688213, 0.9701248, 0.9687972, 0.9701387, -0.0006617, 0.0006584
3: 0.0163371, 0.0259519, 0.0161593, 0.0260542, -0.0048806, 0.0048563
4: -0.0026668, -0.0019356, -0.0026746, -0.0019220, -0.0003694, 0.0003712
5: 0.0145751, 0.0153141, 0.0145672, 0.0153278, -0.0003733, 0.0003752
6: 0.0044580, 0.0048175, 0.0044514, 0.0048213, -0.0001825, 0.0001816
7: -0.0145039, -0.0120122, -0.0145304, -0.0119661, -0.0012586, 0.0012648
8: 0.0052224, 0.0071993, 0.0052014, 0.0072358, -0.0009985, 0.0010035
9: 0.0071177, 0.0106732, 0.0070798, 0.0107389, -0.0017959, 0.0018048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004636
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004573, upper bound: 0.0004702
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040752, -0.0041050, -0.0040752, -0.0000143, 0.0000143
1: -0.0063883, -0.0052704, -0.0063883, -0.0052704, -0.0005348, 0.0005348
2: 0.9687972, 0.9701387, 0.9687972, 0.9701387, -0.0006417, 0.0006417
3: 0.0161593, 0.0260542, 0.0161593, 0.0260542, -0.0047334, 0.0047334
4: -0.0026746, -0.0019220, -0.0026746, -0.0019220, -0.0003600, 0.0003600
5: 0.0145672, 0.0153278, 0.0145672, 0.0153278, -0.0003638, 0.0003638
6: 0.0044514, 0.0048213, 0.0044514, 0.0048213, -0.0001770, 0.0001770
7: -0.0145304, -0.0119661, -0.0145304, -0.0119661, -0.0012267, 0.0012267
8: 0.0052014, 0.0072358, 0.0052014, 0.0072358, -0.0009732, 0.0009732
9: 0.0070798, 0.0107389, 0.0070798, 0.0107389, -0.0017504, 0.0017504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004508
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004573, upper bound: 0.0004569
time: 1.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.32 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004515
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004616, upper bound: 0.0004572
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004556, upper bound: 0.0004506
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004616, upper bound: 0.0004572
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004534, upper bound: 0.0004509
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004573
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004534, upper bound: 0.0004509
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004573
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004646
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004572, upper bound: 0.0004705
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004522
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004572, upper bound: 0.0004575
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004659
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004569, upper bound: 0.0004726
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004516
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004569, upper bound: 0.0004577
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004499
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004569
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004499
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004569
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004502
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004565
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004493
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004565
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004509, upper bound: 0.0004625
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004573, upper bound: 0.0004702
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004509, upper bound: 0.0004505
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004573, upper bound: 0.0004572
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004636
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004573, upper bound: 0.0004702
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004508
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 2, lower bound: -0.0004573, upper bound: 0.0004569

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041045, -0.0040761, -0.0000137, 0.0000129
1: -0.0063904, -0.0053218, -0.0063687, -0.0053048, -0.0005134, 0.0004832
2: 0.9687947, 0.9700771, 0.9688208, 0.9700975, -0.0006160, 0.0005798
3: 0.0161408, 0.0255996, 0.0163330, 0.0257500, -0.0045438, 0.0042765
4: -0.0026400, -0.0019206, -0.0026515, -0.0019352, -0.0003253, 0.0003456
5: 0.0146021, 0.0153292, 0.0145906, 0.0153144, -0.0003287, 0.0003493
6: 0.0044507, 0.0048043, 0.0044578, 0.0048099, -0.0001699, 0.0001599
7: -0.0144126, -0.0119613, -0.0144516, -0.0120111, -0.0011083, 0.0011776
8: 0.0052949, 0.0072396, 0.0052639, 0.0072001, -0.0008793, 0.0009342
9: 0.0072479, 0.0107458, 0.0071923, 0.0106747, -0.0015815, 0.0016803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004542
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004554
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041045, -0.0040759, -0.0000137, 0.0000132
1: -0.0063690, -0.0053059, -0.0063697, -0.0052979, -0.0005148, 0.0004937
2: 0.9688205, 0.9700961, 0.9688196, 0.9701058, -0.0006178, 0.0005924
3: 0.0163308, 0.0257404, 0.0163246, 0.0258114, -0.0045564, 0.0043695
4: -0.0026507, -0.0019351, -0.0026561, -0.0019346, -0.0003323, 0.0003465
5: 0.0145913, 0.0153146, 0.0145858, 0.0153151, -0.0003359, 0.0003502
6: 0.0044578, 0.0048096, 0.0044575, 0.0048122, -0.0001704, 0.0001634
7: -0.0144491, -0.0120105, -0.0144675, -0.0120089, -0.0011324, 0.0011808
8: 0.0052659, 0.0072006, 0.0052513, 0.0072018, -0.0008984, 0.0009368
9: 0.0071958, 0.0106755, 0.0071696, 0.0106778, -0.0016158, 0.0016850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004554, upper bound: 0.0004542
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004554, upper bound: 0.0004616
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041051, -0.0040765, -0.0000136, 0.0000145
1: -0.0063871, -0.0052946, -0.0063904, -0.0053218, -0.0005102, 0.0005442
2: 0.9687987, 0.9701098, 0.9687947, 0.9700771, -0.0006122, 0.0006530
3: 0.0161702, 0.0258404, 0.0161408, 0.0255996, -0.0045158, 0.0048165
4: -0.0026583, -0.0019229, -0.0026400, -0.0019206, -0.0003663, 0.0003435
5: 0.0145836, 0.0153269, 0.0146021, 0.0153292, -0.0003702, 0.0003471
6: 0.0044518, 0.0048133, 0.0044507, 0.0048043, -0.0001688, 0.0001801
7: -0.0144750, -0.0119689, -0.0144126, -0.0119613, -0.0012482, 0.0011703
8: 0.0052454, 0.0072336, 0.0052949, 0.0072396, -0.0009903, 0.0009285
9: 0.0071589, 0.0107349, 0.0072479, 0.0107458, -0.0017811, 0.0016699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004634, upper bound: 0.0004506
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004634, upper bound: 0.0004506
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040756, -0.0041045, -0.0040761, -0.0000139, 0.0000146
1: -0.0063881, -0.0052875, -0.0063690, -0.0053059, -0.0005221, 0.0005450
2: 0.9687974, 0.9701183, 0.9688205, 0.9700961, -0.0006266, 0.0006540
3: 0.0161617, 0.0259031, 0.0163308, 0.0257404, -0.0046215, 0.0048235
4: -0.0026631, -0.0019222, -0.0026507, -0.0019351, -0.0003669, 0.0003515
5: 0.0145788, 0.0153276, 0.0145913, 0.0153146, -0.0003708, 0.0003552
6: 0.0044514, 0.0048157, 0.0044578, 0.0048096, -0.0001728, 0.0001803
7: -0.0144913, -0.0119667, -0.0144491, -0.0120105, -0.0012501, 0.0011977
8: 0.0052325, 0.0072353, 0.0052659, 0.0072006, -0.0009917, 0.0009502
9: 0.0071357, 0.0107380, 0.0071958, 0.0106755, -0.0017837, 0.0017090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004634, upper bound: 0.0004513
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004634, upper bound: 0.0004572
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041044, -0.0040756, -0.0000144, 0.0000130
1: -0.0063904, -0.0053218, -0.0063673, -0.0052888, -0.0005374, 0.0004865
2: 0.9687947, 0.9700771, 0.9688224, 0.9701167, -0.0006449, 0.0005838
3: 0.0161408, 0.0255996, 0.0163451, 0.0258915, -0.0047570, 0.0043058
4: -0.0026400, -0.0019206, -0.0026622, -0.0019362, -0.0003275, 0.0003618
5: 0.0146021, 0.0153292, 0.0145797, 0.0153135, -0.0003310, 0.0003657
6: 0.0044507, 0.0048043, 0.0044583, 0.0048152, -0.0001779, 0.0001610
7: -0.0144126, -0.0119613, -0.0144883, -0.0120142, -0.0011159, 0.0012328
8: 0.0052949, 0.0072396, 0.0052348, 0.0071976, -0.0008853, 0.0009781
9: 0.0072479, 0.0107458, 0.0071400, 0.0106702, -0.0015923, 0.0017591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004534, upper bound: 0.0004542
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004534, upper bound: 0.0004561
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041045, -0.0040755, -0.0000143, 0.0000134
1: -0.0063690, -0.0053059, -0.0063682, -0.0052820, -0.0005374, 0.0005031
2: 0.9688205, 0.9700961, 0.9688213, 0.9701248, -0.0006449, 0.0006038
3: 0.0163308, 0.0257404, 0.0163371, 0.0259519, -0.0047564, 0.0044535
4: -0.0026507, -0.0019351, -0.0026668, -0.0019356, -0.0003387, 0.0003618
5: 0.0145913, 0.0153146, 0.0145751, 0.0153141, -0.0003423, 0.0003656
6: 0.0044578, 0.0048096, 0.0044580, 0.0048175, -0.0001778, 0.0001665
7: -0.0144491, -0.0120105, -0.0145039, -0.0120122, -0.0011542, 0.0012327
8: 0.0052659, 0.0072006, 0.0052224, 0.0071993, -0.0009156, 0.0009779
9: 0.0071958, 0.0106755, 0.0071177, 0.0106732, -0.0016469, 0.0017589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004547, upper bound: 0.0004542
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004547, upper bound: 0.0004628
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041044, -0.0040756, -0.0000151, 0.0000138
1: -0.0064082, -0.0053117, -0.0063673, -0.0052888, -0.0005657, 0.0005173
2: 0.9687734, 0.9700892, 0.9688224, 0.9701167, -0.0006788, 0.0006208
3: 0.0159837, 0.0256892, 0.0163451, 0.0258915, -0.0050069, 0.0045792
4: -0.0026468, -0.0019087, -0.0026622, -0.0019362, -0.0003483, 0.0003808
5: 0.0145952, 0.0153413, 0.0145797, 0.0153135, -0.0003520, 0.0003849
6: 0.0044448, 0.0048077, 0.0044583, 0.0048152, -0.0001872, 0.0001712
7: -0.0144358, -0.0119206, -0.0144883, -0.0120142, -0.0011867, 0.0012976
8: 0.0052764, 0.0072719, 0.0052348, 0.0071976, -0.0009415, 0.0010294
9: 0.0072148, 0.0108039, 0.0071400, 0.0106702, -0.0016934, 0.0018515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004495
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004509
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041045, -0.0040755, -0.0000151, 0.0000143
1: -0.0063873, -0.0052954, -0.0063682, -0.0052820, -0.0005644, 0.0005341
2: 0.9687985, 0.9701087, 0.9688213, 0.9701248, -0.0006774, 0.0006409
3: 0.0161682, 0.0258329, 0.0163371, 0.0259519, -0.0049961, 0.0047271
4: -0.0026578, -0.0019227, -0.0026668, -0.0019356, -0.0003595, 0.0003800
5: 0.0145842, 0.0153271, 0.0145751, 0.0153141, -0.0003634, 0.0003840
6: 0.0044517, 0.0048130, 0.0044580, 0.0048175, -0.0001868, 0.0001767
7: -0.0144731, -0.0119684, -0.0145039, -0.0120122, -0.0012251, 0.0012948
8: 0.0052469, 0.0072340, 0.0052224, 0.0071993, -0.0009719, 0.0010272
9: 0.0071617, 0.0107357, 0.0071177, 0.0106732, -0.0017481, 0.0018475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004635, upper bound: 0.0004495
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004635, upper bound: 0.0004573
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041050, -0.0040758, -0.0000145, 0.0000136
1: -0.0063904, -0.0053218, -0.0063871, -0.0052946, -0.0005442, 0.0005102
2: 0.9687947, 0.9700771, 0.9687987, 0.9701098, -0.0006530, 0.0006122
3: 0.0161408, 0.0255996, 0.0161702, 0.0258404, -0.0048165, 0.0045158
4: -0.0026400, -0.0019206, -0.0026583, -0.0019229, -0.0003435, 0.0003663
5: 0.0146021, 0.0153292, 0.0145836, 0.0153269, -0.0003471, 0.0003702
6: 0.0044507, 0.0048043, 0.0044518, 0.0048133, -0.0001801, 0.0001688
7: -0.0144126, -0.0119613, -0.0144750, -0.0119689, -0.0011703, 0.0012482
8: 0.0052949, 0.0072396, 0.0052454, 0.0072336, -0.0009285, 0.0009903
9: 0.0072479, 0.0107458, 0.0071589, 0.0107349, -0.0016699, 0.0017811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004634
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004643
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041050, -0.0040756, -0.0000146, 0.0000139
1: -0.0063690, -0.0053059, -0.0063881, -0.0052875, -0.0005450, 0.0005221
2: 0.9688205, 0.9700961, 0.9687974, 0.9701183, -0.0006540, 0.0006266
3: 0.0163308, 0.0257404, 0.0161617, 0.0259031, -0.0048235, 0.0046215
4: -0.0026507, -0.0019351, -0.0026631, -0.0019222, -0.0003515, 0.0003669
5: 0.0145913, 0.0153146, 0.0145788, 0.0153276, -0.0003552, 0.0003708
6: 0.0044578, 0.0048096, 0.0044514, 0.0048157, -0.0001803, 0.0001728
7: -0.0144491, -0.0120105, -0.0144913, -0.0119667, -0.0011977, 0.0012501
8: 0.0052659, 0.0072006, 0.0052325, 0.0072353, -0.0009502, 0.0009917
9: 0.0071958, 0.0106755, 0.0071357, 0.0107380, -0.0017090, 0.0017837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004513, upper bound: 0.0004634
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004513, upper bound: 0.0004705
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041050, -0.0040758, -0.0000141, 0.0000133
1: -0.0064082, -0.0053117, -0.0063871, -0.0052946, -0.0005266, 0.0004965
2: 0.9687734, 0.9700892, 0.9687987, 0.9701098, -0.0006319, 0.0005958
3: 0.0159837, 0.0256892, 0.0161702, 0.0258404, -0.0046610, 0.0043943
4: -0.0026468, -0.0019087, -0.0026583, -0.0019229, -0.0003342, 0.0003545
5: 0.0145952, 0.0153413, 0.0145836, 0.0153269, -0.0003378, 0.0003583
6: 0.0044448, 0.0048077, 0.0044518, 0.0048133, -0.0001743, 0.0001643
7: -0.0144358, -0.0119206, -0.0144750, -0.0119689, -0.0011388, 0.0012079
8: 0.0052764, 0.0072719, 0.0052454, 0.0072336, -0.0009035, 0.0009583
9: 0.0072148, 0.0108039, 0.0071589, 0.0107349, -0.0016250, 0.0017236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004637, upper bound: 0.0004513
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004637, upper bound: 0.0004519
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041050, -0.0040756, -0.0000141, 0.0000135
1: -0.0063873, -0.0052954, -0.0063881, -0.0052875, -0.0005280, 0.0005068
2: 0.9687985, 0.9701087, 0.9687974, 0.9701183, -0.0006337, 0.0006082
3: 0.0161682, 0.0258329, 0.0161617, 0.0259031, -0.0046737, 0.0044862
4: -0.0026578, -0.0019227, -0.0026631, -0.0019222, -0.0003412, 0.0003555
5: 0.0145842, 0.0153271, 0.0145788, 0.0153276, -0.0003448, 0.0003593
6: 0.0044517, 0.0048130, 0.0044514, 0.0048157, -0.0001747, 0.0001677
7: -0.0144731, -0.0119684, -0.0144913, -0.0119667, -0.0011626, 0.0012112
8: 0.0052469, 0.0072340, 0.0052325, 0.0072353, -0.0009224, 0.0009609
9: 0.0071617, 0.0107357, 0.0071357, 0.0107380, -0.0016590, 0.0017283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004646, upper bound: 0.0004513
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004646, upper bound: 0.0004575
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041050, -0.0040753, -0.0000150, 0.0000137
1: -0.0063904, -0.0053218, -0.0063874, -0.0052774, -0.0005627, 0.0005113
2: 0.9687947, 0.9700771, 0.9687983, 0.9701304, -0.0006753, 0.0006136
3: 0.0161408, 0.0255996, 0.0161672, 0.0259926, -0.0049807, 0.0045258
4: -0.0026400, -0.0019206, -0.0026699, -0.0019226, -0.0003442, 0.0003788
5: 0.0146021, 0.0153292, 0.0145719, 0.0153272, -0.0003479, 0.0003829
6: 0.0044507, 0.0048043, 0.0044517, 0.0048190, -0.0001862, 0.0001692
7: -0.0144126, -0.0119613, -0.0145145, -0.0119681, -0.0011729, 0.0012908
8: 0.0052949, 0.0072396, 0.0052140, 0.0072342, -0.0009305, 0.0010240
9: 0.0072479, 0.0107458, 0.0071026, 0.0107360, -0.0016736, 0.0018418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004644
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004658
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041050, -0.0040752, -0.0000150, 0.0000141
1: -0.0063690, -0.0053059, -0.0063883, -0.0052704, -0.0005623, 0.0005283
2: 0.9688205, 0.9700961, 0.9687972, 0.9701387, -0.0006748, 0.0006340
3: 0.0163308, 0.0257404, 0.0161593, 0.0260542, -0.0049775, 0.0046761
4: -0.0026507, -0.0019351, -0.0026746, -0.0019220, -0.0003556, 0.0003786
5: 0.0145913, 0.0153146, 0.0145672, 0.0153278, -0.0003594, 0.0003826
6: 0.0044578, 0.0048096, 0.0044514, 0.0048213, -0.0001861, 0.0001748
7: -0.0144491, -0.0120105, -0.0145304, -0.0119661, -0.0012119, 0.0012900
8: 0.0052659, 0.0072006, 0.0052014, 0.0072358, -0.0009614, 0.0010234
9: 0.0071958, 0.0106755, 0.0070798, 0.0107389, -0.0017292, 0.0018407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004644
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004726
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041050, -0.0040753, -0.0000147, 0.0000133
1: -0.0064082, -0.0053117, -0.0063874, -0.0052774, -0.0005510, 0.0004997
2: 0.9687734, 0.9700892, 0.9687983, 0.9701304, -0.0006612, 0.0005996
3: 0.0159837, 0.0256892, 0.0161672, 0.0259926, -0.0048768, 0.0044227
4: -0.0026468, -0.0019087, -0.0026699, -0.0019226, -0.0003364, 0.0003709
5: 0.0145952, 0.0153413, 0.0145719, 0.0153272, -0.0003400, 0.0003749
6: 0.0044448, 0.0048077, 0.0044517, 0.0048190, -0.0001823, 0.0001654
7: -0.0144358, -0.0119206, -0.0145145, -0.0119681, -0.0011462, 0.0012639
8: 0.0052764, 0.0072719, 0.0052140, 0.0072342, -0.0009093, 0.0010027
9: 0.0072148, 0.0108039, 0.0071026, 0.0107360, -0.0016355, 0.0018034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004501
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004514
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041050, -0.0040752, -0.0000147, 0.0000138
1: -0.0063873, -0.0052954, -0.0063883, -0.0052704, -0.0005510, 0.0005162
2: 0.9687985, 0.9701087, 0.9687972, 0.9701387, -0.0006612, 0.0006195
3: 0.0161682, 0.0258329, 0.0161593, 0.0260542, -0.0048767, 0.0045693
4: -0.0026578, -0.0019227, -0.0026746, -0.0019220, -0.0003475, 0.0003709
5: 0.0145842, 0.0153271, 0.0145672, 0.0153278, -0.0003512, 0.0003749
6: 0.0044517, 0.0048130, 0.0044514, 0.0048213, -0.0001823, 0.0001708
7: -0.0144731, -0.0119684, -0.0145304, -0.0119661, -0.0011842, 0.0012638
8: 0.0052469, 0.0072340, 0.0052014, 0.0072358, -0.0009395, 0.0010027
9: 0.0071617, 0.0107357, 0.0070798, 0.0107389, -0.0016897, 0.0018034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004637, upper bound: 0.0004501
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004637, upper bound: 0.0004576
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040756, -0.0041051, -0.0040765, -0.0000130, 0.0000144
1: -0.0063673, -0.0052888, -0.0063904, -0.0053218, -0.0004865, 0.0005374
2: 0.9688224, 0.9701167, 0.9687947, 0.9700771, -0.0005838, 0.0006449
3: 0.0163451, 0.0258915, 0.0161408, 0.0255996, -0.0043058, 0.0047570
4: -0.0026622, -0.0019362, -0.0026400, -0.0019206, -0.0003618, 0.0003275
5: 0.0145797, 0.0153135, 0.0146021, 0.0153292, -0.0003657, 0.0003310
6: 0.0044583, 0.0048152, 0.0044507, 0.0048043, -0.0001610, 0.0001779
7: -0.0144883, -0.0120142, -0.0144126, -0.0119613, -0.0012328, 0.0011159
8: 0.0052348, 0.0071976, 0.0052949, 0.0072396, -0.0009781, 0.0008853
9: 0.0071400, 0.0106702, 0.0072479, 0.0107458, -0.0017591, 0.0015923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004534
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004534
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040755, -0.0041045, -0.0040761, -0.0000134, 0.0000143
1: -0.0063682, -0.0052820, -0.0063690, -0.0053059, -0.0005031, 0.0005374
2: 0.9688213, 0.9701248, 0.9688205, 0.9700961, -0.0006038, 0.0006449
3: 0.0163371, 0.0259519, 0.0163308, 0.0257404, -0.0044535, 0.0047564
4: -0.0026668, -0.0019356, -0.0026507, -0.0019351, -0.0003618, 0.0003387
5: 0.0145751, 0.0153141, 0.0145913, 0.0153146, -0.0003656, 0.0003423
6: 0.0044580, 0.0048175, 0.0044578, 0.0048096, -0.0001665, 0.0001778
7: -0.0145039, -0.0120122, -0.0144491, -0.0120105, -0.0012327, 0.0011542
8: 0.0052224, 0.0071993, 0.0052659, 0.0072006, -0.0009779, 0.0009156
9: 0.0071177, 0.0106732, 0.0071958, 0.0106755, -0.0017589, 0.0016469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004547
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004614
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040753, -0.0041051, -0.0040765, -0.0000137, 0.0000150
1: -0.0063874, -0.0052774, -0.0063904, -0.0053218, -0.0005113, 0.0005627
2: 0.9687983, 0.9701304, 0.9687947, 0.9700771, -0.0006136, 0.0006753
3: 0.0161672, 0.0259926, 0.0161408, 0.0255996, -0.0045258, 0.0049807
4: -0.0026699, -0.0019226, -0.0026400, -0.0019206, -0.0003788, 0.0003442
5: 0.0145719, 0.0153272, 0.0146021, 0.0153292, -0.0003829, 0.0003479
6: 0.0044517, 0.0048190, 0.0044507, 0.0048043, -0.0001692, 0.0001862
7: -0.0145145, -0.0119681, -0.0144126, -0.0119613, -0.0012908, 0.0011729
8: 0.0052140, 0.0072342, 0.0052949, 0.0072396, -0.0010240, 0.0009305
9: 0.0071026, 0.0107360, 0.0072479, 0.0107458, -0.0018418, 0.0016736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004499
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004499
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040752, -0.0041045, -0.0040761, -0.0000141, 0.0000150
1: -0.0063883, -0.0052704, -0.0063690, -0.0053059, -0.0005283, 0.0005623
2: 0.9687972, 0.9701387, 0.9688205, 0.9700961, -0.0006340, 0.0006748
3: 0.0161593, 0.0260542, 0.0163308, 0.0257404, -0.0046761, 0.0049775
4: -0.0026746, -0.0019220, -0.0026507, -0.0019351, -0.0003786, 0.0003556
5: 0.0145672, 0.0153278, 0.0145913, 0.0153146, -0.0003826, 0.0003594
6: 0.0044514, 0.0048213, 0.0044578, 0.0048096, -0.0001748, 0.0001861
7: -0.0145304, -0.0119661, -0.0144491, -0.0120105, -0.0012900, 0.0012119
8: 0.0052014, 0.0072358, 0.0052659, 0.0072006, -0.0010234, 0.0009614
9: 0.0070798, 0.0107389, 0.0071958, 0.0106755, -0.0018407, 0.0017292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004506
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004569
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041044, -0.0040756, -0.0000138, 0.0000130
1: -0.0063893, -0.0053056, -0.0063673, -0.0052888, -0.0005175, 0.0004871
2: 0.9687960, 0.9700965, 0.9688224, 0.9701167, -0.0006210, 0.0005845
3: 0.0161506, 0.0257431, 0.0163451, 0.0258915, -0.0045803, 0.0043115
4: -0.0026509, -0.0019214, -0.0026622, -0.0019362, -0.0003279, 0.0003484
5: 0.0145911, 0.0153285, 0.0145797, 0.0153135, -0.0003314, 0.0003521
6: 0.0044510, 0.0048097, 0.0044583, 0.0048152, -0.0001713, 0.0001612
7: -0.0144498, -0.0119638, -0.0144883, -0.0120142, -0.0011174, 0.0011870
8: 0.0052654, 0.0072376, 0.0052348, 0.0071976, -0.0008865, 0.0009417
9: 0.0071949, 0.0107422, 0.0071400, 0.0106702, -0.0015944, 0.0016938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004531
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004545
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041045, -0.0040755, -0.0000139, 0.0000133
1: -0.0063675, -0.0052901, -0.0063682, -0.0052820, -0.0005196, 0.0004963
2: 0.9688222, 0.9701151, 0.9688213, 0.9701248, -0.0006235, 0.0005955
3: 0.0163433, 0.0258797, 0.0163371, 0.0259519, -0.0045987, 0.0043927
4: -0.0026613, -0.0019360, -0.0026668, -0.0019356, -0.0003341, 0.0003498
5: 0.0145806, 0.0153136, 0.0145751, 0.0153141, -0.0003377, 0.0003535
6: 0.0044582, 0.0048148, 0.0044580, 0.0048175, -0.0001719, 0.0001642
7: -0.0144852, -0.0120138, -0.0145039, -0.0120122, -0.0011384, 0.0011918
8: 0.0052373, 0.0071980, 0.0052224, 0.0071993, -0.0009031, 0.0009455
9: 0.0071444, 0.0106709, 0.0071177, 0.0106732, -0.0016244, 0.0017006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004531
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004612
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040753, -0.0041050, -0.0040761, -0.0000137, 0.0000146
1: -0.0063874, -0.0052774, -0.0063893, -0.0053056, -0.0005142, 0.0005481
2: 0.9687983, 0.9701304, 0.9687960, 0.9700965, -0.0006171, 0.0006577
3: 0.0161672, 0.0259926, 0.0161506, 0.0257431, -0.0045518, 0.0048513
4: -0.0026699, -0.0019226, -0.0026509, -0.0019214, -0.0003690, 0.0003462
5: 0.0145719, 0.0153272, 0.0145911, 0.0153285, -0.0003729, 0.0003499
6: 0.0044517, 0.0048190, 0.0044510, 0.0048097, -0.0001702, 0.0001814
7: -0.0145145, -0.0119681, -0.0144498, -0.0119638, -0.0012573, 0.0011796
8: 0.0052140, 0.0072342, 0.0052654, 0.0072376, -0.0009974, 0.0009359
9: 0.0071026, 0.0107360, 0.0071949, 0.0107422, -0.0017940, 0.0016832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004493
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004493
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040752, -0.0041044, -0.0040757, -0.0000140, 0.0000147
1: -0.0063883, -0.0052704, -0.0063675, -0.0052901, -0.0005249, 0.0005495
2: 0.9687972, 0.9701387, 0.9688222, 0.9701151, -0.0006299, 0.0006595
3: 0.0161593, 0.0260542, 0.0163433, 0.0258797, -0.0046459, 0.0048640
4: -0.0026746, -0.0019220, -0.0026613, -0.0019360, -0.0003699, 0.0003533
5: 0.0145672, 0.0153278, 0.0145806, 0.0153136, -0.0003739, 0.0003571
6: 0.0044514, 0.0048213, 0.0044582, 0.0048148, -0.0001737, 0.0001819
7: -0.0145304, -0.0119661, -0.0144852, -0.0120138, -0.0012606, 0.0012040
8: 0.0052014, 0.0072358, 0.0052373, 0.0071980, -0.0010001, 0.0009552
9: 0.0070798, 0.0107389, 0.0071444, 0.0106709, -0.0017987, 0.0017181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004502
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004565
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040756, -0.0041055, -0.0040763, -0.0000138, 0.0000151
1: -0.0063673, -0.0052888, -0.0064082, -0.0053117, -0.0005173, 0.0005657
2: 0.9688224, 0.9701167, 0.9687734, 0.9700892, -0.0006208, 0.0006788
3: 0.0163451, 0.0258915, 0.0159837, 0.0256892, -0.0045792, 0.0050069
4: -0.0026622, -0.0019362, -0.0026468, -0.0019087, -0.0003808, 0.0003483
5: 0.0145797, 0.0153135, 0.0145952, 0.0153413, -0.0003849, 0.0003520
6: 0.0044583, 0.0048152, 0.0044448, 0.0048077, -0.0001712, 0.0001872
7: -0.0144883, -0.0120142, -0.0144358, -0.0119206, -0.0012976, 0.0011867
8: 0.0052348, 0.0071976, 0.0052764, 0.0072719, -0.0010294, 0.0009415
9: 0.0071400, 0.0106702, 0.0072148, 0.0108039, -0.0018515, 0.0016934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004625
time: 0.84 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004625
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040755, -0.0041050, -0.0040758, -0.0000143, 0.0000151
1: -0.0063682, -0.0052820, -0.0063873, -0.0052954, -0.0005341, 0.0005644
2: 0.9688213, 0.9701248, 0.9687985, 0.9701087, -0.0006409, 0.0006774
3: 0.0163371, 0.0259519, 0.0161682, 0.0258329, -0.0047271, 0.0049961
4: -0.0026668, -0.0019356, -0.0026578, -0.0019227, -0.0003800, 0.0003595
5: 0.0145751, 0.0153141, 0.0145842, 0.0153271, -0.0003840, 0.0003634
6: 0.0044580, 0.0048175, 0.0044517, 0.0048130, -0.0001767, 0.0001868
7: -0.0145039, -0.0120122, -0.0144731, -0.0119684, -0.0012948, 0.0012251
8: 0.0052224, 0.0071993, 0.0052469, 0.0072340, -0.0010272, 0.0009719
9: 0.0071177, 0.0106732, 0.0071617, 0.0107357, -0.0018475, 0.0017481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004635
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004702
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040753, -0.0041055, -0.0040763, -0.0000133, 0.0000147
1: -0.0063874, -0.0052774, -0.0064082, -0.0053117, -0.0004997, 0.0005510
2: 0.9687983, 0.9701304, 0.9687734, 0.9700892, -0.0005996, 0.0006612
3: 0.0161672, 0.0259926, 0.0159837, 0.0256892, -0.0044227, 0.0048768
4: -0.0026699, -0.0019226, -0.0026468, -0.0019087, -0.0003709, 0.0003364
5: 0.0145719, 0.0153272, 0.0145952, 0.0153413, -0.0003749, 0.0003400
6: 0.0044517, 0.0048190, 0.0044448, 0.0048077, -0.0001654, 0.0001823
7: -0.0145145, -0.0119681, -0.0144358, -0.0119206, -0.0012639, 0.0011462
8: 0.0052140, 0.0072342, 0.0052764, 0.0072719, -0.0010027, 0.0009093
9: 0.0071026, 0.0107360, 0.0072148, 0.0108039, -0.0018034, 0.0016355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004505
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004505
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040752, -0.0041050, -0.0040758, -0.0000138, 0.0000147
1: -0.0063883, -0.0052704, -0.0063873, -0.0052954, -0.0005162, 0.0005510
2: 0.9687972, 0.9701387, 0.9687985, 0.9701087, -0.0006195, 0.0006612
3: 0.0161593, 0.0260542, 0.0161682, 0.0258329, -0.0045693, 0.0048767
4: -0.0026746, -0.0019220, -0.0026578, -0.0019227, -0.0003709, 0.0003475
5: 0.0145672, 0.0153278, 0.0145842, 0.0153271, -0.0003749, 0.0003512
6: 0.0044514, 0.0048213, 0.0044517, 0.0048130, -0.0001708, 0.0001823
7: -0.0145304, -0.0119661, -0.0144731, -0.0119684, -0.0012638, 0.0011842
8: 0.0052014, 0.0072358, 0.0052469, 0.0072340, -0.0010027, 0.0009395
9: 0.0070798, 0.0107389, 0.0071617, 0.0107357, -0.0018034, 0.0016897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004512
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004572
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041050, -0.0040753, -0.0000146, 0.0000137
1: -0.0063893, -0.0053056, -0.0063874, -0.0052774, -0.0005481, 0.0005142
2: 0.9687960, 0.9700965, 0.9687983, 0.9701304, -0.0006577, 0.0006171
3: 0.0161506, 0.0257431, 0.0161672, 0.0259926, -0.0048513, 0.0045518
4: -0.0026509, -0.0019214, -0.0026699, -0.0019226, -0.0003462, 0.0003690
5: 0.0145911, 0.0153285, 0.0145719, 0.0153272, -0.0003499, 0.0003729
6: 0.0044510, 0.0048097, 0.0044517, 0.0048190, -0.0001814, 0.0001702
7: -0.0144498, -0.0119638, -0.0145145, -0.0119681, -0.0011796, 0.0012573
8: 0.0052654, 0.0072376, 0.0052140, 0.0072342, -0.0009359, 0.0009974
9: 0.0071949, 0.0107422, 0.0071026, 0.0107360, -0.0016832, 0.0017940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004625
time: 0.84 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004634
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041050, -0.0040752, -0.0000147, 0.0000140
1: -0.0063675, -0.0052901, -0.0063883, -0.0052704, -0.0005495, 0.0005249
2: 0.9688222, 0.9701151, 0.9687972, 0.9701387, -0.0006595, 0.0006299
3: 0.0163433, 0.0258797, 0.0161593, 0.0260542, -0.0048640, 0.0046459
4: -0.0026613, -0.0019360, -0.0026746, -0.0019220, -0.0003533, 0.0003699
5: 0.0145806, 0.0153136, 0.0145672, 0.0153278, -0.0003571, 0.0003739
6: 0.0044582, 0.0048148, 0.0044514, 0.0048213, -0.0001819, 0.0001737
7: -0.0144852, -0.0120138, -0.0145304, -0.0119661, -0.0012040, 0.0012606
8: 0.0052373, 0.0071980, 0.0052014, 0.0072358, -0.0009552, 0.0010001
9: 0.0071444, 0.0106709, 0.0070798, 0.0107389, -0.0017181, 0.0017987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004509, upper bound: 0.0004625
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004509, upper bound: 0.0004702
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040758, -0.0041050, -0.0040753, -0.0000142, 0.0000134
1: -0.0064088, -0.0052944, -0.0063874, -0.0052774, -0.0005309, 0.0005005
2: 0.9687726, 0.9701099, 0.9687983, 0.9701304, -0.0006371, 0.0006006
3: 0.0159781, 0.0258420, 0.0161672, 0.0259926, -0.0046990, 0.0044298
4: -0.0026585, -0.0019083, -0.0026699, -0.0019226, -0.0003369, 0.0003574
5: 0.0145835, 0.0153417, 0.0145719, 0.0153272, -0.0003405, 0.0003612
6: 0.0044446, 0.0048134, 0.0044517, 0.0048190, -0.0001757, 0.0001656
7: -0.0144754, -0.0119191, -0.0145145, -0.0119681, -0.0011480, 0.0012178
8: 0.0052450, 0.0072731, 0.0052140, 0.0072342, -0.0009108, 0.0009661
9: 0.0071583, 0.0108060, 0.0071026, 0.0107360, -0.0016381, 0.0017377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004499
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004507
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041050, -0.0040752, -0.0000142, 0.0000136
1: -0.0063876, -0.0052785, -0.0063883, -0.0052704, -0.0005329, 0.0005100
2: 0.9687980, 0.9701290, 0.9687972, 0.9701387, -0.0006395, 0.0006120
3: 0.0161655, 0.0259825, 0.0161593, 0.0260542, -0.0047168, 0.0045139
4: -0.0026692, -0.0019225, -0.0026746, -0.0019220, -0.0003433, 0.0003587
5: 0.0145727, 0.0153273, 0.0145672, 0.0153278, -0.0003470, 0.0003626
6: 0.0044516, 0.0048186, 0.0044514, 0.0048213, -0.0001764, 0.0001688
7: -0.0145118, -0.0119677, -0.0145304, -0.0119661, -0.0011698, 0.0012224
8: 0.0052161, 0.0072346, 0.0052014, 0.0072358, -0.0009281, 0.0009698
9: 0.0071063, 0.0107367, 0.0070798, 0.0107389, -0.0016692, 0.0017443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004499
time: 0.81 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004568
time: 0.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.11 seconds
IS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004542
IS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004554
IS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004554, upper bound: 0.0004542
IS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004554, upper bound: 0.0004616
IS_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004634, upper bound: 0.0004506
IS_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004634, upper bound: 0.0004506
IS_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004634, upper bound: 0.0004513
IS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004634, upper bound: 0.0004572
IS_A1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004534, upper bound: 0.0004542
IS_A1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004534, upper bound: 0.0004561
IS_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004547, upper bound: 0.0004542
IS_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004547, upper bound: 0.0004628
IS_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004495
IS_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004509
IS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004635, upper bound: 0.0004495
IS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004635, upper bound: 0.0004573
IS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004634
IS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004643
IS_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004513, upper bound: 0.0004634
IS_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004513, upper bound: 0.0004705
IS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004637, upper bound: 0.0004513
IS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004637, upper bound: 0.0004519
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004646, upper bound: 0.0004513
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004646, upper bound: 0.0004575
IS_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004644
IS_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004658
IS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004644
IS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004506, upper bound: 0.0004726
IS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004501
IS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004628, upper bound: 0.0004514
IS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004637, upper bound: 0.0004501
IS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004637, upper bound: 0.0004576
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004534
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004534
IS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004547
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004614
IS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004499
IS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004499
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004506
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004569
IS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004531
IS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004542, upper bound: 0.0004545
IS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004531
IS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004561, upper bound: 0.0004612
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004493
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004493
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004502
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004565
IS_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004625
IS_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004625
IS_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004635
IS_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004702
IS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004505
IS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004505
IS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004512
IS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004572
IS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004625
IS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004634
IS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004509, upper bound: 0.0004625
IS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004509, upper bound: 0.0004702
IS_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004499
IS_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004647, upper bound: 0.0004507
IS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004499
IS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004568

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041051, -0.0040765, -0.0000131, 0.0000131
1: -0.0063904, -0.0053218, -0.0063904, -0.0053218, -0.0004924, 0.0004924
2: 0.9687947, 0.9700771, 0.9687947, 0.9700771, -0.0005909, 0.0005909
3: 0.0161408, 0.0255996, 0.0161408, 0.0255996, -0.0043584, 0.0043584
4: -0.0026400, -0.0019206, -0.0026400, -0.0019206, -0.0003315, 0.0003315
5: 0.0146021, 0.0153292, 0.0146021, 0.0153292, -0.0003350, 0.0003350
6: 0.0044507, 0.0048043, 0.0044507, 0.0048043, -0.0001630, 0.0001630
7: -0.0144126, -0.0119613, -0.0144126, -0.0119613, -0.0011295, 0.0011295
8: 0.0052949, 0.0072396, 0.0052949, 0.0072396, -0.0008961, 0.0008961
9: 0.0072479, 0.0107458, 0.0072479, 0.0107458, -0.0016117, 0.0016117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004330
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041045, -0.0040761, -0.0000139, 0.0000129
1: -0.0063904, -0.0053218, -0.0063690, -0.0053059, -0.0005193, 0.0004827
2: 0.9687947, 0.9700771, 0.9688205, 0.9700961, -0.0006231, 0.0005793
3: 0.0161408, 0.0255996, 0.0163308, 0.0257404, -0.0045961, 0.0042730
4: -0.0026400, -0.0019206, -0.0026507, -0.0019351, -0.0003250, 0.0003496
5: 0.0146021, 0.0153292, 0.0145913, 0.0153146, -0.0003285, 0.0003533
6: 0.0044507, 0.0048043, 0.0044578, 0.0048096, -0.0001718, 0.0001598
7: -0.0144126, -0.0119613, -0.0144491, -0.0120105, -0.0011074, 0.0011911
8: 0.0052949, 0.0072396, 0.0052659, 0.0072006, -0.0008785, 0.0009450
9: 0.0072479, 0.0107458, 0.0071958, 0.0106755, -0.0015801, 0.0016996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004385
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004305
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041051, -0.0040765, -0.0000129, 0.0000139
1: -0.0063690, -0.0053059, -0.0063904, -0.0053218, -0.0004827, 0.0005193
2: 0.9688205, 0.9700961, 0.9687947, 0.9700771, -0.0005793, 0.0006231
3: 0.0163308, 0.0257404, 0.0161408, 0.0255996, -0.0042730, 0.0045961
4: -0.0026507, -0.0019351, -0.0026400, -0.0019206, -0.0003496, 0.0003250
5: 0.0145913, 0.0153146, 0.0146021, 0.0153292, -0.0003533, 0.0003285
6: 0.0044578, 0.0048096, 0.0044507, 0.0048043, -0.0001598, 0.0001718
7: -0.0144491, -0.0120105, -0.0144126, -0.0119613, -0.0011911, 0.0011074
8: 0.0052659, 0.0072006, 0.0052949, 0.0072396, -0.0009450, 0.0008785
9: 0.0071958, 0.0106755, 0.0072479, 0.0107458, -0.0016996, 0.0015801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004385, upper bound: 0.0004284
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041045, -0.0040761, -0.0000131, 0.0000131
1: -0.0063690, -0.0053059, -0.0063690, -0.0053059, -0.0004922, 0.0004922
2: 0.9688205, 0.9700961, 0.9688205, 0.9700961, -0.0005906, 0.0005906
3: 0.0163308, 0.0257404, 0.0163308, 0.0257404, -0.0043562, 0.0043562
4: -0.0026507, -0.0019351, -0.0026507, -0.0019351, -0.0003313, 0.0003313
5: 0.0145913, 0.0153146, 0.0145913, 0.0153146, -0.0003349, 0.0003349
6: 0.0044578, 0.0048096, 0.0044578, 0.0048096, -0.0001629, 0.0001629
7: -0.0144491, -0.0120105, -0.0144491, -0.0120105, -0.0011290, 0.0011290
8: 0.0052659, 0.0072006, 0.0052659, 0.0072006, -0.0008957, 0.0008957
9: 0.0071958, 0.0106755, 0.0071958, 0.0106755, -0.0016109, 0.0016109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004307, upper bound: 0.0004519
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004509
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041051, -0.0040765, -0.0000139, 0.0000140
1: -0.0064082, -0.0053117, -0.0063904, -0.0053218, -0.0005206, 0.0005233
2: 0.9687734, 0.9700892, 0.9687947, 0.9700771, -0.0006248, 0.0006280
3: 0.0159837, 0.0256892, 0.0161408, 0.0255996, -0.0046084, 0.0046319
4: -0.0026468, -0.0019087, -0.0026400, -0.0019206, -0.0003523, 0.0003505
5: 0.0145952, 0.0153413, 0.0146021, 0.0153292, -0.0003560, 0.0003542
6: 0.0044448, 0.0048077, 0.0044507, 0.0048043, -0.0001723, 0.0001732
7: -0.0144358, -0.0119206, -0.0144126, -0.0119613, -0.0012004, 0.0011943
8: 0.0052764, 0.0072719, 0.0052949, 0.0072396, -0.0009523, 0.0009475
9: 0.0072148, 0.0108039, 0.0072479, 0.0107458, -0.0017129, 0.0017042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004440, upper bound: 0.0004229
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041051, -0.0040765, -0.0000136, 0.0000146
1: -0.0063873, -0.0052954, -0.0063904, -0.0053218, -0.0005098, 0.0005484
2: 0.9687985, 0.9701087, 0.9687947, 0.9700771, -0.0006118, 0.0006581
3: 0.0161682, 0.0258329, 0.0161408, 0.0255996, -0.0045126, 0.0048540
4: -0.0026578, -0.0019227, -0.0026400, -0.0019206, -0.0003692, 0.0003432
5: 0.0145842, 0.0153271, 0.0146021, 0.0153292, -0.0003731, 0.0003469
6: 0.0044517, 0.0048130, 0.0044507, 0.0048043, -0.0001687, 0.0001815
7: -0.0144731, -0.0119684, -0.0144126, -0.0119613, -0.0012580, 0.0011695
8: 0.0052469, 0.0072340, 0.0052949, 0.0072396, -0.0009980, 0.0009278
9: 0.0071617, 0.0107357, 0.0072479, 0.0107458, -0.0017950, 0.0016688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004440, upper bound: 0.0004258
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004254
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041045, -0.0040761, -0.0000146, 0.0000137
1: -0.0064082, -0.0053117, -0.0063690, -0.0053059, -0.0005475, 0.0005136
2: 0.9687734, 0.9700892, 0.9688205, 0.9700961, -0.0006570, 0.0006164
3: 0.0159837, 0.0256892, 0.0163308, 0.0257404, -0.0048460, 0.0045464
4: -0.0026468, -0.0019087, -0.0026507, -0.0019351, -0.0003458, 0.0003686
5: 0.0145952, 0.0153413, 0.0145913, 0.0153146, -0.0003495, 0.0003725
6: 0.0044448, 0.0048077, 0.0044578, 0.0048096, -0.0001812, 0.0001700
7: -0.0144358, -0.0119206, -0.0144491, -0.0120105, -0.0011782, 0.0012559
8: 0.0052764, 0.0072719, 0.0052659, 0.0072006, -0.0009348, 0.0009964
9: 0.0072148, 0.0108039, 0.0071958, 0.0106755, -0.0016812, 0.0017921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004440, upper bound: 0.0004289
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004288
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041045, -0.0040761, -0.0000139, 0.0000140
1: -0.0063873, -0.0052954, -0.0063690, -0.0053059, -0.0005206, 0.0005231
2: 0.9687985, 0.9701087, 0.9688205, 0.9700961, -0.0006248, 0.0006277
3: 0.0161682, 0.0258329, 0.0163308, 0.0257404, -0.0046082, 0.0046299
4: -0.0026578, -0.0019227, -0.0026507, -0.0019351, -0.0003521, 0.0003505
5: 0.0145842, 0.0153271, 0.0145913, 0.0153146, -0.0003559, 0.0003542
6: 0.0044517, 0.0048130, 0.0044578, 0.0048096, -0.0001723, 0.0001731
7: -0.0144731, -0.0119684, -0.0144491, -0.0120105, -0.0011999, 0.0011943
8: 0.0052469, 0.0072340, 0.0052659, 0.0072006, -0.0009519, 0.0009475
9: 0.0071617, 0.0107357, 0.0071958, 0.0106755, -0.0017121, 0.0017041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004440, upper bound: 0.0004468
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004468
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041050, -0.0040761, -0.0000138, 0.0000134
1: -0.0063904, -0.0053218, -0.0063893, -0.0053056, -0.0005185, 0.0005018
2: 0.9687947, 0.9700771, 0.9687960, 0.9700965, -0.0006222, 0.0006022
3: 0.0161408, 0.0255996, 0.0161506, 0.0257431, -0.0045894, 0.0044420
4: -0.0026400, -0.0019206, -0.0026509, -0.0019214, -0.0003378, 0.0003491
5: 0.0146021, 0.0153292, 0.0145911, 0.0153285, -0.0003414, 0.0003528
6: 0.0044507, 0.0048043, 0.0044510, 0.0048097, -0.0001716, 0.0001661
7: -0.0144126, -0.0119613, -0.0144498, -0.0119638, -0.0011512, 0.0011894
8: 0.0052949, 0.0072396, 0.0052654, 0.0072376, -0.0009133, 0.0009436
9: 0.0072479, 0.0107458, 0.0071949, 0.0107422, -0.0016426, 0.0016972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004193, upper bound: 0.0004284
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041044, -0.0040757, -0.0000144, 0.0000130
1: -0.0063904, -0.0053218, -0.0063675, -0.0052901, -0.0005402, 0.0004863
2: 0.9687947, 0.9700771, 0.9688222, 0.9701151, -0.0006483, 0.0005836
3: 0.0161408, 0.0255996, 0.0163433, 0.0258797, -0.0047819, 0.0043043
4: -0.0026400, -0.0019206, -0.0026613, -0.0019360, -0.0003274, 0.0003637
5: 0.0146021, 0.0153292, 0.0145806, 0.0153136, -0.0003309, 0.0003676
6: 0.0044507, 0.0048043, 0.0044582, 0.0048148, -0.0001788, 0.0001609
7: -0.0144126, -0.0119613, -0.0144852, -0.0120138, -0.0011155, 0.0012393
8: 0.0052949, 0.0072396, 0.0052373, 0.0071980, -0.0008850, 0.0009832
9: 0.0072479, 0.0107458, 0.0071444, 0.0106709, -0.0015917, 0.0017683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004193, upper bound: 0.0004364
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004261
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041050, -0.0040761, -0.0000136, 0.0000141
1: -0.0063690, -0.0053059, -0.0063893, -0.0053056, -0.0005088, 0.0005287
2: 0.9688205, 0.9700961, 0.9687960, 0.9700965, -0.0006106, 0.0006345
3: 0.0163308, 0.0257404, 0.0161506, 0.0257431, -0.0045040, 0.0046796
4: -0.0026507, -0.0019351, -0.0026509, -0.0019214, -0.0003559, 0.0003426
5: 0.0145913, 0.0153146, 0.0145911, 0.0153285, -0.0003597, 0.0003462
6: 0.0044578, 0.0048096, 0.0044510, 0.0048097, -0.0001684, 0.0001750
7: -0.0144491, -0.0120105, -0.0144498, -0.0119638, -0.0012128, 0.0011672
8: 0.0052659, 0.0072006, 0.0052654, 0.0072376, -0.0009622, 0.0009260
9: 0.0071958, 0.0106755, 0.0071949, 0.0107422, -0.0017305, 0.0016656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004352, upper bound: 0.0004231
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041044, -0.0040757, -0.0000139, 0.0000134
1: -0.0063690, -0.0053059, -0.0063675, -0.0052901, -0.0005187, 0.0005017
2: 0.9688205, 0.9700961, 0.9688222, 0.9701151, -0.0006224, 0.0006021
3: 0.0163308, 0.0257404, 0.0163433, 0.0258797, -0.0045908, 0.0044410
4: -0.0026507, -0.0019351, -0.0026613, -0.0019360, -0.0003378, 0.0003492
5: 0.0145913, 0.0153146, 0.0145806, 0.0153136, -0.0003414, 0.0003529
6: 0.0044578, 0.0048096, 0.0044582, 0.0048148, -0.0001716, 0.0001660
7: -0.0144491, -0.0120105, -0.0144852, -0.0120138, -0.0011509, 0.0011897
8: 0.0052659, 0.0072006, 0.0052373, 0.0071980, -0.0009131, 0.0009439
9: 0.0071958, 0.0106755, 0.0071444, 0.0106709, -0.0016423, 0.0016977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004265, upper bound: 0.0004533
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004505
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041050, -0.0040761, -0.0000146, 0.0000142
1: -0.0064082, -0.0053117, -0.0063893, -0.0053056, -0.0005467, 0.0005327
2: 0.9687734, 0.9700892, 0.9687960, 0.9700965, -0.0006561, 0.0006393
3: 0.0159837, 0.0256892, 0.0161506, 0.0257431, -0.0048394, 0.0047154
4: -0.0026468, -0.0019087, -0.0026509, -0.0019214, -0.0003586, 0.0003681
5: 0.0145952, 0.0153413, 0.0145911, 0.0153285, -0.0003625, 0.0003720
6: 0.0044448, 0.0048077, 0.0044510, 0.0048097, -0.0001809, 0.0001763
7: -0.0144358, -0.0119206, -0.0144498, -0.0119638, -0.0012220, 0.0012542
8: 0.0052764, 0.0072719, 0.0052654, 0.0072376, -0.0009695, 0.0009950
9: 0.0072148, 0.0108039, 0.0071949, 0.0107422, -0.0017437, 0.0017896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004387, upper bound: 0.0004152
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041044, -0.0040757, -0.0000152, 0.0000138
1: -0.0064082, -0.0053117, -0.0063675, -0.0052901, -0.0005685, 0.0005172
2: 0.9687734, 0.9700892, 0.9688222, 0.9701151, -0.0006822, 0.0006206
3: 0.0159837, 0.0256892, 0.0163433, 0.0258797, -0.0050318, 0.0045777
4: -0.0026468, -0.0019087, -0.0026613, -0.0019360, -0.0003482, 0.0003827
5: 0.0145952, 0.0153413, 0.0145806, 0.0153136, -0.0003519, 0.0003868
6: 0.0044448, 0.0048077, 0.0044582, 0.0048148, -0.0001881, 0.0001712
7: -0.0144358, -0.0119206, -0.0144852, -0.0120138, -0.0011864, 0.0013040
8: 0.0052764, 0.0072719, 0.0052373, 0.0071980, -0.0009412, 0.0010346
9: 0.0072148, 0.0108039, 0.0071444, 0.0106709, -0.0016928, 0.0018608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004276, upper bound: 0.0004333
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004241
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041050, -0.0040761, -0.0000143, 0.0000149
1: -0.0063873, -0.0052954, -0.0063893, -0.0053056, -0.0005359, 0.0005578
2: 0.9687985, 0.9701087, 0.9687960, 0.9700965, -0.0006431, 0.0006694
3: 0.0161682, 0.0258329, 0.0161506, 0.0257431, -0.0047436, 0.0049375
4: -0.0026578, -0.0019227, -0.0026509, -0.0019214, -0.0003755, 0.0003608
5: 0.0145842, 0.0153271, 0.0145911, 0.0153285, -0.0003795, 0.0003646
6: 0.0044517, 0.0048130, 0.0044510, 0.0048097, -0.0001774, 0.0001846
7: -0.0144731, -0.0119684, -0.0144498, -0.0119638, -0.0012796, 0.0012293
8: 0.0052469, 0.0072340, 0.0052654, 0.0072376, -0.0010152, 0.0009753
9: 0.0071617, 0.0107357, 0.0071949, 0.0107422, -0.0018259, 0.0017542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004190
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041044, -0.0040757, -0.0000146, 0.0000142
1: -0.0063873, -0.0052954, -0.0063675, -0.0052901, -0.0005471, 0.0005326
2: 0.9687985, 0.9701087, 0.9688222, 0.9701151, -0.0006566, 0.0006392
3: 0.0161682, 0.0258329, 0.0163433, 0.0258797, -0.0048427, 0.0047146
4: -0.0026578, -0.0019227, -0.0026613, -0.0019360, -0.0003586, 0.0003683
5: 0.0145842, 0.0153271, 0.0145806, 0.0153136, -0.0003624, 0.0003722
6: 0.0044517, 0.0048130, 0.0044582, 0.0048148, -0.0001811, 0.0001763
7: -0.0144731, -0.0119684, -0.0144852, -0.0120138, -0.0012218, 0.0012550
8: 0.0052469, 0.0072340, 0.0052373, 0.0071980, -0.0009693, 0.0009957
9: 0.0071617, 0.0107357, 0.0071444, 0.0106709, -0.0017435, 0.0017908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004455
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004456
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041055, -0.0040763, -0.0000140, 0.0000139
1: -0.0063904, -0.0053218, -0.0064082, -0.0053117, -0.0005233, 0.0005206
2: 0.9687947, 0.9700771, 0.9687734, 0.9700892, -0.0006280, 0.0006248
3: 0.0161408, 0.0255996, 0.0159837, 0.0256892, -0.0046319, 0.0046084
4: -0.0026400, -0.0019206, -0.0026468, -0.0019087, -0.0003505, 0.0003523
5: 0.0146021, 0.0153292, 0.0145952, 0.0153413, -0.0003542, 0.0003560
6: 0.0044507, 0.0048043, 0.0044448, 0.0048077, -0.0001732, 0.0001723
7: -0.0144126, -0.0119613, -0.0144358, -0.0119206, -0.0011943, 0.0012004
8: 0.0052949, 0.0072396, 0.0052764, 0.0072719, -0.0009475, 0.0009523
9: 0.0072479, 0.0107458, 0.0072148, 0.0108039, -0.0017042, 0.0017129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004229, upper bound: 0.0004440
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041050, -0.0040758, -0.0000146, 0.0000136
1: -0.0063904, -0.0053218, -0.0063873, -0.0052954, -0.0005484, 0.0005098
2: 0.9687947, 0.9700771, 0.9687985, 0.9701087, -0.0006581, 0.0006118
3: 0.0161408, 0.0255996, 0.0161682, 0.0258329, -0.0048540, 0.0045126
4: -0.0026400, -0.0019206, -0.0026578, -0.0019227, -0.0003432, 0.0003692
5: 0.0146021, 0.0153292, 0.0145842, 0.0153271, -0.0003469, 0.0003731
6: 0.0044507, 0.0048043, 0.0044517, 0.0048130, -0.0001815, 0.0001687
7: -0.0144126, -0.0119613, -0.0144731, -0.0119684, -0.0011695, 0.0012580
8: 0.0052949, 0.0072396, 0.0052469, 0.0072340, -0.0009278, 0.0009980
9: 0.0072479, 0.0107458, 0.0071617, 0.0107357, -0.0016688, 0.0017950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004229, upper bound: 0.0004489
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004408
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041055, -0.0040763, -0.0000137, 0.0000146
1: -0.0063690, -0.0053059, -0.0064082, -0.0053117, -0.0005136, 0.0005475
2: 0.9688205, 0.9700961, 0.9687734, 0.9700892, -0.0006164, 0.0006570
3: 0.0163308, 0.0257404, 0.0159837, 0.0256892, -0.0045464, 0.0048460
4: -0.0026507, -0.0019351, -0.0026468, -0.0019087, -0.0003686, 0.0003458
5: 0.0145913, 0.0153146, 0.0145952, 0.0153413, -0.0003725, 0.0003495
6: 0.0044578, 0.0048096, 0.0044448, 0.0048077, -0.0001700, 0.0001812
7: -0.0144491, -0.0120105, -0.0144358, -0.0119206, -0.0012559, 0.0011782
8: 0.0052659, 0.0072006, 0.0052764, 0.0072719, -0.0009964, 0.0009348
9: 0.0071958, 0.0106755, 0.0072148, 0.0108039, -0.0017921, 0.0016812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004289, upper bound: 0.0004450
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041050, -0.0040758, -0.0000140, 0.0000139
1: -0.0063690, -0.0053059, -0.0063873, -0.0052954, -0.0005231, 0.0005206
2: 0.9688205, 0.9700961, 0.9687985, 0.9701087, -0.0006277, 0.0006248
3: 0.0163308, 0.0257404, 0.0161682, 0.0258329, -0.0046299, 0.0046082
4: -0.0026507, -0.0019351, -0.0026578, -0.0019227, -0.0003505, 0.0003521
5: 0.0145913, 0.0153146, 0.0145842, 0.0153271, -0.0003542, 0.0003559
6: 0.0044578, 0.0048096, 0.0044517, 0.0048130, -0.0001731, 0.0001723
7: -0.0144491, -0.0120105, -0.0144731, -0.0119684, -0.0011943, 0.0011999
8: 0.0052659, 0.0072006, 0.0052469, 0.0072340, -0.0009475, 0.0009519
9: 0.0071958, 0.0106755, 0.0071617, 0.0107357, -0.0017041, 0.0017121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004289, upper bound: 0.0004613
time: 1.01 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004592
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041055, -0.0040763, -0.0000135, 0.0000135
1: -0.0064082, -0.0053117, -0.0064082, -0.0053117, -0.0005056, 0.0005056
2: 0.9687734, 0.9700892, 0.9687734, 0.9700892, -0.0006068, 0.0006068
3: 0.0159837, 0.0256892, 0.0159837, 0.0256892, -0.0044756, 0.0044756
4: -0.0026468, -0.0019087, -0.0026468, -0.0019087, -0.0003404, 0.0003404
5: 0.0145952, 0.0153413, 0.0145952, 0.0153413, -0.0003440, 0.0003440
6: 0.0044448, 0.0048077, 0.0044448, 0.0048077, -0.0001673, 0.0001673
7: -0.0144358, -0.0119206, -0.0144358, -0.0119206, -0.0011599, 0.0011599
8: 0.0052764, 0.0072719, 0.0052764, 0.0072719, -0.0009202, 0.0009202
9: 0.0072148, 0.0108039, 0.0072148, 0.0108039, -0.0016551, 0.0016551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004341, upper bound: 0.0004321
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041050, -0.0040758, -0.0000142, 0.0000132
1: -0.0064082, -0.0053117, -0.0063873, -0.0052954, -0.0005324, 0.0004960
2: 0.9687734, 0.9700892, 0.9687985, 0.9701087, -0.0006389, 0.0005953
3: 0.0159837, 0.0256892, 0.0161682, 0.0258329, -0.0047123, 0.0043907
4: -0.0026468, -0.0019087, -0.0026578, -0.0019227, -0.0003339, 0.0003584
5: 0.0145952, 0.0153413, 0.0145842, 0.0153271, -0.0003375, 0.0003622
6: 0.0044448, 0.0048077, 0.0044517, 0.0048130, -0.0001762, 0.0001642
7: -0.0144358, -0.0119206, -0.0144731, -0.0119684, -0.0011379, 0.0012212
8: 0.0052764, 0.0072719, 0.0052469, 0.0072340, -0.0009027, 0.0009689
9: 0.0072148, 0.0108039, 0.0071617, 0.0107357, -0.0016237, 0.0017426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004341, upper bound: 0.0004372
time: 1.13 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004301
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041055, -0.0040763, -0.0000132, 0.0000142
1: -0.0063873, -0.0052954, -0.0064082, -0.0053117, -0.0004960, 0.0005324
2: 0.9687985, 0.9701087, 0.9687734, 0.9700892, -0.0005953, 0.0006389
3: 0.0161682, 0.0258329, 0.0159837, 0.0256892, -0.0043907, 0.0047123
4: -0.0026578, -0.0019227, -0.0026468, -0.0019087, -0.0003584, 0.0003339
5: 0.0145842, 0.0153271, 0.0145952, 0.0153413, -0.0003622, 0.0003375
6: 0.0044517, 0.0048130, 0.0044448, 0.0048077, -0.0001642, 0.0001762
7: -0.0144731, -0.0119684, -0.0144358, -0.0119206, -0.0012212, 0.0011379
8: 0.0052469, 0.0072340, 0.0052764, 0.0072719, -0.0009689, 0.0009027
9: 0.0071617, 0.0107357, 0.0072148, 0.0108039, -0.0017426, 0.0016237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004493, upper bound: 0.0004270
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041050, -0.0040758, -0.0000135, 0.0000135
1: -0.0063873, -0.0052954, -0.0063873, -0.0052954, -0.0005054, 0.0005054
2: 0.9687985, 0.9701087, 0.9687985, 0.9701087, -0.0006065, 0.0006065
3: 0.0161682, 0.0258329, 0.0161682, 0.0258329, -0.0044731, 0.0044731
4: -0.0026578, -0.0019227, -0.0026578, -0.0019227, -0.0003402, 0.0003402
5: 0.0145842, 0.0153271, 0.0145842, 0.0153271, -0.0003438, 0.0003438
6: 0.0044517, 0.0048130, 0.0044517, 0.0048130, -0.0001672, 0.0001672
7: -0.0144731, -0.0119684, -0.0144731, -0.0119684, -0.0011592, 0.0011592
8: 0.0052469, 0.0072340, 0.0052469, 0.0072340, -0.0009197, 0.0009197
9: 0.0071617, 0.0107357, 0.0071617, 0.0107357, -0.0016541, 0.0016541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004412, upper bound: 0.0004485
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004472
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041056, -0.0040758, -0.0000145, 0.0000141
1: -0.0063904, -0.0053218, -0.0064088, -0.0052944, -0.0005437, 0.0005268
2: 0.9687947, 0.9700771, 0.9687726, 0.9701099, -0.0006524, 0.0006322
3: 0.0161408, 0.0255996, 0.0159781, 0.0258420, -0.0048120, 0.0046631
4: -0.0026400, -0.0019206, -0.0026585, -0.0019083, -0.0003547, 0.0003660
5: 0.0146021, 0.0153292, 0.0145835, 0.0153417, -0.0003584, 0.0003699
6: 0.0044507, 0.0048043, 0.0044446, 0.0048134, -0.0001799, 0.0001743
7: -0.0144126, -0.0119613, -0.0144754, -0.0119191, -0.0012085, 0.0012471
8: 0.0052949, 0.0072396, 0.0052450, 0.0072731, -0.0009588, 0.0009894
9: 0.0072479, 0.0107458, 0.0071583, 0.0108060, -0.0017244, 0.0017795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004176, upper bound: 0.0004400
time: 0.87 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041051, -0.0040765, -0.0041050, -0.0040754, -0.0000151, 0.0000136
1: -0.0063904, -0.0053218, -0.0063876, -0.0052785, -0.0005646, 0.0005111
2: 0.9687947, 0.9700771, 0.9687980, 0.9701290, -0.0006775, 0.0006134
3: 0.0161408, 0.0255996, 0.0161655, 0.0259825, -0.0049971, 0.0045242
4: -0.0026400, -0.0019206, -0.0026692, -0.0019225, -0.0003441, 0.0003801
5: 0.0146021, 0.0153292, 0.0145727, 0.0153273, -0.0003478, 0.0003841
6: 0.0044507, 0.0048043, 0.0044516, 0.0048186, -0.0001868, 0.0001692
7: -0.0144126, -0.0119613, -0.0145118, -0.0119677, -0.0011725, 0.0012950
8: 0.0052949, 0.0072396, 0.0052161, 0.0072346, -0.0009302, 0.0010274
9: 0.0072479, 0.0107458, 0.0071063, 0.0107367, -0.0016730, 0.0018479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004176, upper bound: 0.0004476
time: 1.07 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004371
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041056, -0.0040758, -0.0000143, 0.0000148
1: -0.0063690, -0.0053059, -0.0064088, -0.0052944, -0.0005340, 0.0005537
2: 0.9688205, 0.9700961, 0.9687726, 0.9701099, -0.0006408, 0.0006644
3: 0.0163308, 0.0257404, 0.0159781, 0.0258420, -0.0047266, 0.0049008
4: -0.0026507, -0.0019351, -0.0026585, -0.0019083, -0.0003727, 0.0003595
5: 0.0145913, 0.0153146, 0.0145835, 0.0153417, -0.0003767, 0.0003633
6: 0.0044578, 0.0048096, 0.0044446, 0.0048134, -0.0001767, 0.0001832
7: -0.0144491, -0.0120105, -0.0144754, -0.0119191, -0.0012701, 0.0012249
8: 0.0052659, 0.0072006, 0.0052450, 0.0072731, -0.0010076, 0.0009718
9: 0.0071958, 0.0106755, 0.0071583, 0.0108060, -0.0018123, 0.0017479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004250, upper bound: 0.0004424
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041050, -0.0040754, -0.0000145, 0.0000141
1: -0.0063690, -0.0053059, -0.0063876, -0.0052785, -0.0005441, 0.0005269
2: 0.9688205, 0.9700961, 0.9687980, 0.9701290, -0.0006530, 0.0006322
3: 0.0163308, 0.0257404, 0.0161655, 0.0259825, -0.0048163, 0.0046633
4: -0.0026507, -0.0019351, -0.0026692, -0.0019225, -0.0003547, 0.0003663
5: 0.0145913, 0.0153146, 0.0145727, 0.0153273, -0.0003585, 0.0003702
6: 0.0044578, 0.0048096, 0.0044516, 0.0048186, -0.0001801, 0.0001744
7: -0.0144491, -0.0120105, -0.0145118, -0.0119677, -0.0012085, 0.0012482
8: 0.0052659, 0.0072006, 0.0052161, 0.0072346, -0.0009588, 0.0009903
9: 0.0071958, 0.0106755, 0.0071063, 0.0107367, -0.0017245, 0.0017811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004250, upper bound: 0.0004632
time: 1.06 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004606
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041056, -0.0040758, -0.0000142, 0.0000138
1: -0.0064082, -0.0053117, -0.0064088, -0.0052944, -0.0005322, 0.0005150
2: 0.9687734, 0.9700892, 0.9687726, 0.9701099, -0.0006387, 0.0006180
3: 0.0159837, 0.0256892, 0.0159781, 0.0258420, -0.0047106, 0.0045584
4: -0.0026468, -0.0019087, -0.0026585, -0.0019083, -0.0003467, 0.0003583
5: 0.0145952, 0.0153413, 0.0145835, 0.0153417, -0.0003504, 0.0003621
6: 0.0044448, 0.0048077, 0.0044446, 0.0048134, -0.0001761, 0.0001704
7: -0.0144358, -0.0119206, -0.0144754, -0.0119191, -0.0011813, 0.0012208
8: 0.0052764, 0.0072719, 0.0052450, 0.0072731, -0.0009372, 0.0009685
9: 0.0072148, 0.0108039, 0.0071583, 0.0108060, -0.0016857, 0.0017420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004280, upper bound: 0.0004276
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041050, -0.0040754, -0.0000148, 0.0000133
1: -0.0064082, -0.0053117, -0.0063876, -0.0052785, -0.0005537, 0.0004995
2: 0.9687734, 0.9700892, 0.9687980, 0.9701290, -0.0006645, 0.0005994
3: 0.0159837, 0.0256892, 0.0161655, 0.0259825, -0.0049014, 0.0044210
4: -0.0026468, -0.0019087, -0.0026692, -0.0019225, -0.0003362, 0.0003728
5: 0.0145952, 0.0153413, 0.0145727, 0.0153273, -0.0003398, 0.0003768
6: 0.0044448, 0.0048077, 0.0044516, 0.0048186, -0.0001833, 0.0001653
7: -0.0144358, -0.0119206, -0.0145118, -0.0119677, -0.0011457, 0.0012702
8: 0.0052764, 0.0072719, 0.0052161, 0.0072346, -0.0009090, 0.0010077
9: 0.0072148, 0.0108039, 0.0071063, 0.0107367, -0.0016349, 0.0018125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004280, upper bound: 0.0004346
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004254
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041056, -0.0040758, -0.0000140, 0.0000145
1: -0.0063873, -0.0052954, -0.0064088, -0.0052944, -0.0005226, 0.0005417
2: 0.9687985, 0.9701087, 0.9687726, 0.9701099, -0.0006271, 0.0006501
3: 0.0161682, 0.0258329, 0.0159781, 0.0258420, -0.0046257, 0.0047951
4: -0.0026578, -0.0019227, -0.0026585, -0.0019083, -0.0003647, 0.0003518
5: 0.0145842, 0.0153271, 0.0145835, 0.0153417, -0.0003686, 0.0003556
6: 0.0044517, 0.0048130, 0.0044446, 0.0048134, -0.0001729, 0.0001793
7: -0.0144731, -0.0119684, -0.0144754, -0.0119191, -0.0012427, 0.0011988
8: 0.0052469, 0.0072340, 0.0052450, 0.0072731, -0.0009859, 0.0009511
9: 0.0071617, 0.0107357, 0.0071583, 0.0108060, -0.0017732, 0.0017106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004453, upper bound: 0.0004203
time: 0.88 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041050, -0.0040754, -0.0000142, 0.0000137
1: -0.0063873, -0.0052954, -0.0063876, -0.0052785, -0.0005323, 0.0005148
2: 0.9687985, 0.9701087, 0.9687980, 0.9701290, -0.0006387, 0.0006178
3: 0.0161682, 0.0258329, 0.0161655, 0.0259825, -0.0047112, 0.0045568
4: -0.0026578, -0.0019227, -0.0026692, -0.0019225, -0.0003466, 0.0003583
5: 0.0145842, 0.0153271, 0.0145727, 0.0153273, -0.0003503, 0.0003621
6: 0.0044517, 0.0048130, 0.0044516, 0.0048186, -0.0001761, 0.0001704
7: -0.0144731, -0.0119684, -0.0145118, -0.0119677, -0.0011809, 0.0012210
8: 0.0052469, 0.0072340, 0.0052161, 0.0072346, -0.0009369, 0.0009686
9: 0.0071617, 0.0107357, 0.0071063, 0.0107367, -0.0016851, 0.0017422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004363, upper bound: 0.0004482
time: 0.92 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004460
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041051, -0.0040765, -0.0000134, 0.0000138
1: -0.0063893, -0.0053056, -0.0063904, -0.0053218, -0.0005018, 0.0005185
2: 0.9687960, 0.9700965, 0.9687947, 0.9700771, -0.0006022, 0.0006222
3: 0.0161506, 0.0257431, 0.0161408, 0.0255996, -0.0044420, 0.0045894
4: -0.0026509, -0.0019214, -0.0026400, -0.0019206, -0.0003491, 0.0003378
5: 0.0145911, 0.0153285, 0.0146021, 0.0153292, -0.0003528, 0.0003414
6: 0.0044510, 0.0048097, 0.0044507, 0.0048043, -0.0001661, 0.0001716
7: -0.0144498, -0.0119638, -0.0144126, -0.0119613, -0.0011894, 0.0011512
8: 0.0052654, 0.0072376, 0.0052949, 0.0072396, -0.0009436, 0.0009133
9: 0.0071949, 0.0107422, 0.0072479, 0.0107458, -0.0016972, 0.0016426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004193
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004162
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041051, -0.0040765, -0.0000130, 0.0000144
1: -0.0063675, -0.0052901, -0.0063904, -0.0053218, -0.0004863, 0.0005402
2: 0.9688222, 0.9701151, 0.9687947, 0.9700771, -0.0005836, 0.0006483
3: 0.0163433, 0.0258797, 0.0161408, 0.0255996, -0.0043043, 0.0047819
4: -0.0026613, -0.0019360, -0.0026400, -0.0019206, -0.0003637, 0.0003274
5: 0.0145806, 0.0153136, 0.0146021, 0.0153292, -0.0003676, 0.0003309
6: 0.0044582, 0.0048148, 0.0044507, 0.0048043, -0.0001609, 0.0001788
7: -0.0144852, -0.0120138, -0.0144126, -0.0119613, -0.0012393, 0.0011155
8: 0.0052373, 0.0071980, 0.0052949, 0.0072396, -0.0009832, 0.0008850
9: 0.0071444, 0.0106709, 0.0072479, 0.0107458, -0.0017683, 0.0015917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004245
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004232
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041045, -0.0040761, -0.0000141, 0.0000136
1: -0.0063893, -0.0053056, -0.0063690, -0.0053059, -0.0005287, 0.0005088
2: 0.9687960, 0.9700965, 0.9688205, 0.9700961, -0.0006345, 0.0006106
3: 0.0161506, 0.0257431, 0.0163308, 0.0257404, -0.0046796, 0.0045040
4: -0.0026509, -0.0019214, -0.0026507, -0.0019351, -0.0003426, 0.0003559
5: 0.0145911, 0.0153285, 0.0145913, 0.0153146, -0.0003462, 0.0003597
6: 0.0044510, 0.0048097, 0.0044578, 0.0048096, -0.0001750, 0.0001684
7: -0.0144498, -0.0119638, -0.0144491, -0.0120105, -0.0011672, 0.0012128
8: 0.0052654, 0.0072376, 0.0052659, 0.0072006, -0.0009260, 0.0009622
9: 0.0071949, 0.0107422, 0.0071958, 0.0106755, -0.0016656, 0.0017305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004181, upper bound: 0.0004352
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004257
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041045, -0.0040761, -0.0000134, 0.0000139
1: -0.0063675, -0.0052901, -0.0063690, -0.0053059, -0.0005017, 0.0005187
2: 0.9688222, 0.9701151, 0.9688205, 0.9700961, -0.0006021, 0.0006224
3: 0.0163433, 0.0258797, 0.0163308, 0.0257404, -0.0044410, 0.0045908
4: -0.0026613, -0.0019360, -0.0026507, -0.0019351, -0.0003492, 0.0003378
5: 0.0145806, 0.0153136, 0.0145913, 0.0153146, -0.0003529, 0.0003414
6: 0.0044582, 0.0048148, 0.0044578, 0.0048096, -0.0001660, 0.0001716
7: -0.0144852, -0.0120138, -0.0144491, -0.0120105, -0.0011897, 0.0011509
8: 0.0052373, 0.0071980, 0.0052659, 0.0072006, -0.0009439, 0.0009131
9: 0.0071444, 0.0106709, 0.0071958, 0.0106755, -0.0016977, 0.0016423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004505
time: 1.03 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004505
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040758, -0.0041051, -0.0040765, -0.0000141, 0.0000145
1: -0.0064088, -0.0052944, -0.0063904, -0.0053218, -0.0005268, 0.0005437
2: 0.9687726, 0.9701099, 0.9687947, 0.9700771, -0.0006322, 0.0006524
3: 0.0159781, 0.0258420, 0.0161408, 0.0255996, -0.0046631, 0.0048121
4: -0.0026585, -0.0019083, -0.0026400, -0.0019206, -0.0003660, 0.0003547
5: 0.0145835, 0.0153417, 0.0146021, 0.0153292, -0.0003699, 0.0003584
6: 0.0044446, 0.0048134, 0.0044507, 0.0048043, -0.0001743, 0.0001799
7: -0.0144754, -0.0119191, -0.0144126, -0.0119613, -0.0012471, 0.0012085
8: 0.0052450, 0.0072731, 0.0052949, 0.0072396, -0.0009894, 0.0009588
9: 0.0071583, 0.0108060, 0.0072479, 0.0107458, -0.0017795, 0.0017244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004177
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004150
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041051, -0.0040765, -0.0000136, 0.0000151
1: -0.0063876, -0.0052785, -0.0063904, -0.0053218, -0.0005111, 0.0005646
2: 0.9687980, 0.9701290, 0.9687947, 0.9700771, -0.0006134, 0.0006775
3: 0.0161655, 0.0259825, 0.0161408, 0.0255996, -0.0045242, 0.0049971
4: -0.0026692, -0.0019225, -0.0026400, -0.0019206, -0.0003801, 0.0003441
5: 0.0145727, 0.0153273, 0.0146021, 0.0153292, -0.0003841, 0.0003478
6: 0.0044516, 0.0048186, 0.0044507, 0.0048043, -0.0001692, 0.0001868
7: -0.0145118, -0.0119677, -0.0144126, -0.0119613, -0.0012950, 0.0011725
8: 0.0052161, 0.0072346, 0.0052949, 0.0072396, -0.0010274, 0.0009302
9: 0.0071063, 0.0107367, 0.0072479, 0.0107458, -0.0018479, 0.0016730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004219
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004210
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040758, -0.0041045, -0.0040761, -0.0000148, 0.0000143
1: -0.0064088, -0.0052944, -0.0063690, -0.0053059, -0.0005537, 0.0005340
2: 0.9687726, 0.9701099, 0.9688205, 0.9700961, -0.0006644, 0.0006408
3: 0.0159781, 0.0258420, 0.0163308, 0.0257404, -0.0049008, 0.0047266
4: -0.0026585, -0.0019083, -0.0026507, -0.0019351, -0.0003595, 0.0003727
5: 0.0145835, 0.0153417, 0.0145913, 0.0153146, -0.0003633, 0.0003767
6: 0.0044446, 0.0048134, 0.0044578, 0.0048096, -0.0001832, 0.0001767
7: -0.0144754, -0.0119191, -0.0144491, -0.0120105, -0.0012249, 0.0012701
8: 0.0052450, 0.0072731, 0.0052659, 0.0072006, -0.0009718, 0.0010076
9: 0.0071583, 0.0108060, 0.0071958, 0.0106755, -0.0017479, 0.0018123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004176
time: 1.06 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004245
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041045, -0.0040761, -0.0000141, 0.0000145
1: -0.0063876, -0.0052785, -0.0063690, -0.0053059, -0.0005269, 0.0005441
2: 0.9687980, 0.9701290, 0.9688205, 0.9700961, -0.0006322, 0.0006530
3: 0.0161655, 0.0259825, 0.0163308, 0.0257404, -0.0046633, 0.0048163
4: -0.0026692, -0.0019225, -0.0026507, -0.0019351, -0.0003663, 0.0003547
5: 0.0145727, 0.0153273, 0.0145913, 0.0153146, -0.0003702, 0.0003585
6: 0.0044516, 0.0048186, 0.0044578, 0.0048096, -0.0001744, 0.0001801
7: -0.0145118, -0.0119677, -0.0144491, -0.0120105, -0.0012482, 0.0012085
8: 0.0052161, 0.0072346, 0.0052659, 0.0072006, -0.0009903, 0.0009588
9: 0.0071063, 0.0107367, 0.0071958, 0.0106755, -0.0017811, 0.0017245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004464
time: 0.93 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004464
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041050, -0.0040761, -0.0000132, 0.0000132
1: -0.0063893, -0.0053056, -0.0063893, -0.0053056, -0.0004958, 0.0004958
2: 0.9687960, 0.9700965, 0.9687960, 0.9700965, -0.0005950, 0.0005950
3: 0.0161506, 0.0257431, 0.0161506, 0.0257431, -0.0043889, 0.0043889
4: -0.0026509, -0.0019214, -0.0026509, -0.0019214, -0.0003338, 0.0003338
5: 0.0145911, 0.0153285, 0.0145911, 0.0153285, -0.0003374, 0.0003374
6: 0.0044510, 0.0048097, 0.0044510, 0.0048097, -0.0001641, 0.0001641
7: -0.0144498, -0.0119638, -0.0144498, -0.0119638, -0.0011374, 0.0011374
8: 0.0052654, 0.0072376, 0.0052654, 0.0072376, -0.0009024, 0.0009024
9: 0.0071949, 0.0107422, 0.0071949, 0.0107422, -0.0016230, 0.0016230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004279
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004140, upper bound: 0.0004140
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041044, -0.0040757, -0.0000140, 0.0000130
1: -0.0063893, -0.0053056, -0.0063675, -0.0052901, -0.0005237, 0.0004866
2: 0.9687960, 0.9700965, 0.9688222, 0.9701151, -0.0006284, 0.0005839
3: 0.0161506, 0.0257431, 0.0163433, 0.0258797, -0.0046353, 0.0043071
4: -0.0026509, -0.0019214, -0.0026613, -0.0019360, -0.0003276, 0.0003525
5: 0.0145911, 0.0153285, 0.0145806, 0.0153136, -0.0003311, 0.0003563
6: 0.0044510, 0.0048097, 0.0044582, 0.0048148, -0.0001733, 0.0001610
7: -0.0144498, -0.0119638, -0.0144852, -0.0120138, -0.0011162, 0.0012013
8: 0.0052654, 0.0072376, 0.0052373, 0.0071980, -0.0008856, 0.0009530
9: 0.0071949, 0.0107422, 0.0071444, 0.0106709, -0.0015928, 0.0017141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004351
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004140, upper bound: 0.0004254
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041050, -0.0040761, -0.0000130, 0.0000140
1: -0.0063675, -0.0052901, -0.0063893, -0.0053056, -0.0004866, 0.0005237
2: 0.9688222, 0.9701151, 0.9687960, 0.9700965, -0.0005839, 0.0006284
3: 0.0163433, 0.0258797, 0.0161506, 0.0257431, -0.0043071, 0.0046353
4: -0.0026613, -0.0019360, -0.0026509, -0.0019214, -0.0003525, 0.0003276
5: 0.0145806, 0.0153136, 0.0145911, 0.0153285, -0.0003563, 0.0003311
6: 0.0044582, 0.0048148, 0.0044510, 0.0048097, -0.0001610, 0.0001733
7: -0.0144852, -0.0120138, -0.0144498, -0.0119638, -0.0012013, 0.0011162
8: 0.0052373, 0.0071980, 0.0052654, 0.0072376, -0.0009530, 0.0008856
9: 0.0071444, 0.0106709, 0.0071949, 0.0107422, -0.0017141, 0.0015928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004364, upper bound: 0.0004231
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004261, upper bound: 0.0004214
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041044, -0.0040757, -0.0000132, 0.0000132
1: -0.0063675, -0.0052901, -0.0063675, -0.0052901, -0.0004947, 0.0004947
2: 0.9688222, 0.9701151, 0.9688222, 0.9701151, -0.0005937, 0.0005937
3: 0.0163433, 0.0258797, 0.0163433, 0.0258797, -0.0043787, 0.0043787
4: -0.0026613, -0.0019360, -0.0026613, -0.0019360, -0.0003330, 0.0003330
5: 0.0145806, 0.0153136, 0.0145806, 0.0153136, -0.0003366, 0.0003366
6: 0.0044582, 0.0048148, 0.0044582, 0.0048148, -0.0001637, 0.0001637
7: -0.0144852, -0.0120138, -0.0144852, -0.0120138, -0.0011348, 0.0011348
8: 0.0052373, 0.0071980, 0.0052373, 0.0071980, -0.0009003, 0.0009003
9: 0.0071444, 0.0106709, 0.0071444, 0.0106709, -0.0016192, 0.0016192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004264, upper bound: 0.0004514
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004261, upper bound: 0.0004499
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040758, -0.0041050, -0.0040761, -0.0000140, 0.0000141
1: -0.0064088, -0.0052944, -0.0063893, -0.0053056, -0.0005242, 0.0005269
2: 0.9687726, 0.9701099, 0.9687960, 0.9700965, -0.0006290, 0.0006323
3: 0.0159781, 0.0258420, 0.0161506, 0.0257431, -0.0046395, 0.0046634
4: -0.0026585, -0.0019083, -0.0026509, -0.0019214, -0.0003547, 0.0003529
5: 0.0145835, 0.0153417, 0.0145911, 0.0153285, -0.0003585, 0.0003566
6: 0.0044446, 0.0048134, 0.0044510, 0.0048097, -0.0001735, 0.0001744
7: -0.0144754, -0.0119191, -0.0144498, -0.0119638, -0.0012086, 0.0012024
8: 0.0052450, 0.0072731, 0.0052654, 0.0072376, -0.0009588, 0.0009539
9: 0.0071583, 0.0108060, 0.0071949, 0.0107422, -0.0017245, 0.0017157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004152
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004126
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041050, -0.0040761, -0.0000137, 0.0000148
1: -0.0063876, -0.0052785, -0.0063893, -0.0053056, -0.0005139, 0.0005528
2: 0.9687980, 0.9701290, 0.9687960, 0.9700965, -0.0006167, 0.0006634
3: 0.0161655, 0.0259825, 0.0161506, 0.0257431, -0.0045484, 0.0048928
4: -0.0026692, -0.0019225, -0.0026509, -0.0019214, -0.0003721, 0.0003459
5: 0.0145727, 0.0153273, 0.0145911, 0.0153285, -0.0003761, 0.0003496
6: 0.0044516, 0.0048186, 0.0044510, 0.0048097, -0.0001701, 0.0001829
7: -0.0145118, -0.0119677, -0.0144498, -0.0119638, -0.0012680, 0.0011788
8: 0.0052161, 0.0072346, 0.0052654, 0.0072376, -0.0010060, 0.0009352
9: 0.0071063, 0.0107367, 0.0071949, 0.0107422, -0.0018093, 0.0016820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004190
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004182
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040758, -0.0041044, -0.0040757, -0.0000147, 0.0000138
1: -0.0064088, -0.0052944, -0.0063675, -0.0052901, -0.0005520, 0.0005176
2: 0.9687726, 0.9701099, 0.9688222, 0.9701151, -0.0006624, 0.0006212
3: 0.0159781, 0.0258420, 0.0163433, 0.0258797, -0.0048859, 0.0045816
4: -0.0026585, -0.0019083, -0.0026613, -0.0019360, -0.0003485, 0.0003716
5: 0.0145835, 0.0153417, 0.0145806, 0.0153136, -0.0003522, 0.0003756
6: 0.0044446, 0.0048134, 0.0044582, 0.0048148, -0.0001827, 0.0001713
7: -0.0144754, -0.0119191, -0.0144852, -0.0120138, -0.0011874, 0.0012662
8: 0.0052450, 0.0072731, 0.0052373, 0.0071980, -0.0009420, 0.0010046
9: 0.0071583, 0.0108060, 0.0071444, 0.0106709, -0.0016943, 0.0018068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004243
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004239
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041044, -0.0040757, -0.0000140, 0.0000140
1: -0.0063876, -0.0052785, -0.0063675, -0.0052901, -0.0005233, 0.0005258
2: 0.9687980, 0.9701290, 0.9688222, 0.9701151, -0.0006280, 0.0006310
3: 0.0161655, 0.0259825, 0.0163433, 0.0258797, -0.0046321, 0.0046542
4: -0.0026692, -0.0019225, -0.0026613, -0.0019360, -0.0003540, 0.0003523
5: 0.0145727, 0.0153273, 0.0145806, 0.0153136, -0.0003578, 0.0003561
6: 0.0044516, 0.0048186, 0.0044582, 0.0048148, -0.0001732, 0.0001740
7: -0.0145118, -0.0119677, -0.0144852, -0.0120138, -0.0012062, 0.0012005
8: 0.0052161, 0.0072346, 0.0052373, 0.0071980, -0.0009569, 0.0009524
9: 0.0071063, 0.0107367, 0.0071444, 0.0106709, -0.0017211, 0.0017130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004455
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004455
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041055, -0.0040763, -0.0000142, 0.0000146
1: -0.0063893, -0.0053056, -0.0064082, -0.0053117, -0.0005327, 0.0005467
2: 0.9687960, 0.9700965, 0.9687734, 0.9700892, -0.0006393, 0.0006561
3: 0.0161506, 0.0257431, 0.0159837, 0.0256892, -0.0047154, 0.0048394
4: -0.0026509, -0.0019214, -0.0026468, -0.0019087, -0.0003681, 0.0003586
5: 0.0145911, 0.0153285, 0.0145952, 0.0153413, -0.0003720, 0.0003625
6: 0.0044510, 0.0048097, 0.0044448, 0.0048077, -0.0001763, 0.0001809
7: -0.0144498, -0.0119638, -0.0144358, -0.0119206, -0.0012542, 0.0012220
8: 0.0052654, 0.0072376, 0.0052764, 0.0072719, -0.0009950, 0.0009695
9: 0.0071949, 0.0107422, 0.0072148, 0.0108039, -0.0017896, 0.0017437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004387
time: 0.82 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004245
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041055, -0.0040763, -0.0000138, 0.0000152
1: -0.0063675, -0.0052901, -0.0064082, -0.0053117, -0.0005172, 0.0005685
2: 0.9688222, 0.9701151, 0.9687734, 0.9700892, -0.0006206, 0.0006822
3: 0.0163433, 0.0258797, 0.0159837, 0.0256892, -0.0045777, 0.0050318
4: -0.0026613, -0.0019360, -0.0026468, -0.0019087, -0.0003827, 0.0003482
5: 0.0145806, 0.0153136, 0.0145952, 0.0153413, -0.0003868, 0.0003519
6: 0.0044582, 0.0048148, 0.0044448, 0.0048077, -0.0001712, 0.0001881
7: -0.0144852, -0.0120138, -0.0144358, -0.0119206, -0.0013040, 0.0011864
8: 0.0052373, 0.0071980, 0.0052764, 0.0072719, -0.0010346, 0.0009412
9: 0.0071444, 0.0106709, 0.0072148, 0.0108039, -0.0018608, 0.0016928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004258, upper bound: 0.0004329
time: 1.02 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004316
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041050, -0.0040758, -0.0000149, 0.0000143
1: -0.0063893, -0.0053056, -0.0063873, -0.0052954, -0.0005578, 0.0005359
2: 0.9687960, 0.9700965, 0.9687985, 0.9701087, -0.0006694, 0.0006431
3: 0.0161506, 0.0257431, 0.0161682, 0.0258329, -0.0049375, 0.0047436
4: -0.0026509, -0.0019214, -0.0026578, -0.0019227, -0.0003608, 0.0003755
5: 0.0145911, 0.0153285, 0.0145842, 0.0153271, -0.0003646, 0.0003795
6: 0.0044510, 0.0048097, 0.0044517, 0.0048130, -0.0001846, 0.0001774
7: -0.0144498, -0.0119638, -0.0144731, -0.0119684, -0.0012293, 0.0012796
8: 0.0052654, 0.0072376, 0.0052469, 0.0072340, -0.0009753, 0.0010152
9: 0.0071949, 0.0107422, 0.0071617, 0.0107357, -0.0017542, 0.0018259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004450
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004357
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041050, -0.0040758, -0.0000142, 0.0000146
1: -0.0063675, -0.0052901, -0.0063873, -0.0052954, -0.0005326, 0.0005471
2: 0.9688222, 0.9701151, 0.9687985, 0.9701087, -0.0006392, 0.0006566
3: 0.0163433, 0.0258797, 0.0161682, 0.0258329, -0.0047146, 0.0048427
4: -0.0026613, -0.0019360, -0.0026578, -0.0019227, -0.0003683, 0.0003586
5: 0.0145806, 0.0153136, 0.0145842, 0.0153271, -0.0003722, 0.0003624
6: 0.0044582, 0.0048148, 0.0044517, 0.0048130, -0.0001763, 0.0001811
7: -0.0144852, -0.0120138, -0.0144731, -0.0119684, -0.0012550, 0.0012218
8: 0.0052373, 0.0071980, 0.0052469, 0.0072340, -0.0009957, 0.0009693
9: 0.0071444, 0.0106709, 0.0071617, 0.0107357, -0.0017908, 0.0017435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004609
time: 1.01 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004588
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040758, -0.0041055, -0.0040763, -0.0000138, 0.0000142
1: -0.0064088, -0.0052944, -0.0064082, -0.0053117, -0.0005150, 0.0005322
2: 0.9687726, 0.9701099, 0.9687734, 0.9700892, -0.0006180, 0.0006387
3: 0.0159781, 0.0258420, 0.0159837, 0.0256892, -0.0045584, 0.0047106
4: -0.0026585, -0.0019083, -0.0026468, -0.0019087, -0.0003583, 0.0003467
5: 0.0145835, 0.0153417, 0.0145952, 0.0153413, -0.0003621, 0.0003504
6: 0.0044446, 0.0048134, 0.0044448, 0.0048077, -0.0001704, 0.0001761
7: -0.0144754, -0.0119191, -0.0144358, -0.0119206, -0.0012208, 0.0011814
8: 0.0052450, 0.0072731, 0.0052764, 0.0072719, -0.0009685, 0.0009372
9: 0.0071583, 0.0108060, 0.0072148, 0.0108039, -0.0017420, 0.0016857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004406, upper bound: 0.0004189
time: 0.83 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004162
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041055, -0.0040763, -0.0000133, 0.0000148
1: -0.0063876, -0.0052785, -0.0064082, -0.0053117, -0.0004995, 0.0005537
2: 0.9687980, 0.9701290, 0.9687734, 0.9700892, -0.0005994, 0.0006645
3: 0.0161655, 0.0259825, 0.0159837, 0.0256892, -0.0044210, 0.0049014
4: -0.0026692, -0.0019225, -0.0026468, -0.0019087, -0.0003728, 0.0003362
5: 0.0145727, 0.0153273, 0.0145952, 0.0153413, -0.0003768, 0.0003398
6: 0.0044516, 0.0048186, 0.0044448, 0.0048077, -0.0001653, 0.0001833
7: -0.0145118, -0.0119677, -0.0144358, -0.0119206, -0.0012702, 0.0011457
8: 0.0052161, 0.0072346, 0.0052764, 0.0072719, -0.0010077, 0.0009090
9: 0.0071063, 0.0107367, 0.0072148, 0.0108039, -0.0018125, 0.0016349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004406, upper bound: 0.0004231
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004222
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040758, -0.0041050, -0.0040758, -0.0000145, 0.0000140
1: -0.0064088, -0.0052944, -0.0063873, -0.0052954, -0.0005417, 0.0005226
2: 0.9687726, 0.9701099, 0.9687985, 0.9701087, -0.0006501, 0.0006271
3: 0.0159781, 0.0258420, 0.0161682, 0.0258329, -0.0047951, 0.0046257
4: -0.0026585, -0.0019083, -0.0026578, -0.0019227, -0.0003518, 0.0003647
5: 0.0145835, 0.0153417, 0.0145842, 0.0153271, -0.0003556, 0.0003686
6: 0.0044446, 0.0048134, 0.0044517, 0.0048130, -0.0001793, 0.0001729
7: -0.0144754, -0.0119191, -0.0144731, -0.0119684, -0.0011988, 0.0012427
8: 0.0052450, 0.0072731, 0.0052469, 0.0072340, -0.0009511, 0.0009859
9: 0.0071583, 0.0108060, 0.0071617, 0.0107357, -0.0017106, 0.0017732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004342
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004257
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041050, -0.0040758, -0.0000137, 0.0000142
1: -0.0063876, -0.0052785, -0.0063873, -0.0052954, -0.0005148, 0.0005323
2: 0.9687980, 0.9701290, 0.9687985, 0.9701087, -0.0006178, 0.0006387
3: 0.0161655, 0.0259825, 0.0161682, 0.0258329, -0.0045568, 0.0047112
4: -0.0026692, -0.0019225, -0.0026578, -0.0019227, -0.0003583, 0.0003466
5: 0.0145727, 0.0153273, 0.0145842, 0.0153271, -0.0003621, 0.0003503
6: 0.0044516, 0.0048186, 0.0044517, 0.0048130, -0.0001704, 0.0001761
7: -0.0145118, -0.0119677, -0.0144731, -0.0119684, -0.0012210, 0.0011809
8: 0.0052161, 0.0072346, 0.0052469, 0.0072340, -0.0009686, 0.0009369
9: 0.0071063, 0.0107367, 0.0071617, 0.0107357, -0.0017422, 0.0016851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004406, upper bound: 0.0004467
time: 1.07 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004467
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041056, -0.0040758, -0.0000141, 0.0000140
1: -0.0063893, -0.0053056, -0.0064088, -0.0052944, -0.0005269, 0.0005242
2: 0.9687960, 0.9700965, 0.9687726, 0.9701099, -0.0006323, 0.0006290
3: 0.0161506, 0.0257431, 0.0159781, 0.0258420, -0.0046634, 0.0046395
4: -0.0026509, -0.0019214, -0.0026585, -0.0019083, -0.0003529, 0.0003547
5: 0.0145911, 0.0153285, 0.0145835, 0.0153417, -0.0003566, 0.0003585
6: 0.0044510, 0.0048097, 0.0044446, 0.0048134, -0.0001744, 0.0001735
7: -0.0144498, -0.0119638, -0.0144754, -0.0119191, -0.0012024, 0.0012086
8: 0.0052654, 0.0072376, 0.0052450, 0.0072731, -0.0009539, 0.0009588
9: 0.0071949, 0.0107422, 0.0071583, 0.0108060, -0.0017157, 0.0017245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004387
time: 0.86 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004126, upper bound: 0.0004243
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041050, -0.0040754, -0.0000148, 0.0000137
1: -0.0063893, -0.0053056, -0.0063876, -0.0052785, -0.0005528, 0.0005139
2: 0.9687960, 0.9700965, 0.9687980, 0.9701290, -0.0006634, 0.0006167
3: 0.0161506, 0.0257431, 0.0161655, 0.0259825, -0.0048928, 0.0045484
4: -0.0026509, -0.0019214, -0.0026692, -0.0019225, -0.0003459, 0.0003721
5: 0.0145911, 0.0153285, 0.0145727, 0.0153273, -0.0003496, 0.0003761
6: 0.0044510, 0.0048097, 0.0044516, 0.0048186, -0.0001829, 0.0001701
7: -0.0144498, -0.0119638, -0.0145118, -0.0119677, -0.0011788, 0.0012680
8: 0.0052654, 0.0072376, 0.0052161, 0.0072346, -0.0009352, 0.0010060
9: 0.0071949, 0.0107422, 0.0071063, 0.0107367, -0.0016820, 0.0018093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004450
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004126, upper bound: 0.0004357
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041056, -0.0040758, -0.0000138, 0.0000147
1: -0.0063675, -0.0052901, -0.0064088, -0.0052944, -0.0005176, 0.0005520
2: 0.9688222, 0.9701151, 0.9687726, 0.9701099, -0.0006212, 0.0006624
3: 0.0163433, 0.0258797, 0.0159781, 0.0258420, -0.0045816, 0.0048859
4: -0.0026613, -0.0019360, -0.0026585, -0.0019083, -0.0003716, 0.0003485
5: 0.0145806, 0.0153136, 0.0145835, 0.0153417, -0.0003756, 0.0003522
6: 0.0044582, 0.0048148, 0.0044446, 0.0048134, -0.0001713, 0.0001827
7: -0.0144852, -0.0120138, -0.0144754, -0.0119191, -0.0012662, 0.0011874
8: 0.0052373, 0.0071980, 0.0052450, 0.0072731, -0.0010046, 0.0009420
9: 0.0071444, 0.0106709, 0.0071583, 0.0108060, -0.0018068, 0.0016943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004246, upper bound: 0.0004413
time: 0.82 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004241, upper bound: 0.0004311
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040757, -0.0041050, -0.0040754, -0.0000140, 0.0000140
1: -0.0063675, -0.0052901, -0.0063876, -0.0052785, -0.0005258, 0.0005233
2: 0.9688222, 0.9701151, 0.9687980, 0.9701290, -0.0006310, 0.0006280
3: 0.0163433, 0.0258797, 0.0161655, 0.0259825, -0.0046542, 0.0046321
4: -0.0026613, -0.0019360, -0.0026692, -0.0019225, -0.0003523, 0.0003540
5: 0.0145806, 0.0153136, 0.0145727, 0.0153273, -0.0003561, 0.0003578
6: 0.0044582, 0.0048148, 0.0044516, 0.0048186, -0.0001740, 0.0001732
7: -0.0144852, -0.0120138, -0.0145118, -0.0119677, -0.0012005, 0.0012062
8: 0.0052373, 0.0071980, 0.0052161, 0.0072346, -0.0009524, 0.0009569
9: 0.0071444, 0.0106709, 0.0071063, 0.0107367, -0.0017130, 0.0017211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004246, upper bound: 0.0004609
time: 1.01 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004241, upper bound: 0.0004587
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040758, -0.0041056, -0.0040758, -0.0000136, 0.0000136
1: -0.0064088, -0.0052944, -0.0064088, -0.0052944, -0.0005093, 0.0005093
2: 0.9687726, 0.9701099, 0.9687726, 0.9701099, -0.0006112, 0.0006112
3: 0.0159781, 0.0258420, 0.0159781, 0.0258420, -0.0045078, 0.0045078
4: -0.0026585, -0.0019083, -0.0026585, -0.0019083, -0.0003428, 0.0003428
5: 0.0145835, 0.0153417, 0.0145835, 0.0153417, -0.0003465, 0.0003465
6: 0.0044446, 0.0048134, 0.0044446, 0.0048134, -0.0001685, 0.0001685
7: -0.0144754, -0.0119191, -0.0144754, -0.0119191, -0.0011682, 0.0011682
8: 0.0052450, 0.0072731, 0.0052450, 0.0072731, -0.0009268, 0.0009268
9: 0.0071583, 0.0108060, 0.0071583, 0.0108060, -0.0016670, 0.0016670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004283, upper bound: 0.0004275
time: 0.86 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004142
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040758, -0.0041050, -0.0040754, -0.0000143, 0.0000134
1: -0.0064088, -0.0052944, -0.0063876, -0.0052785, -0.0005370, 0.0005000
2: 0.9687726, 0.9701099, 0.9687980, 0.9701290, -0.0006444, 0.0006000
3: 0.0159781, 0.0258420, 0.0161655, 0.0259825, -0.0047530, 0.0044257
4: -0.0026585, -0.0019083, -0.0026692, -0.0019225, -0.0003366, 0.0003615
5: 0.0145835, 0.0153417, 0.0145727, 0.0153273, -0.0003402, 0.0003654
6: 0.0044446, 0.0048134, 0.0044516, 0.0048186, -0.0001777, 0.0001655
7: -0.0144754, -0.0119191, -0.0145118, -0.0119677, -0.0011470, 0.0012318
8: 0.0052450, 0.0072731, 0.0052161, 0.0072346, -0.0009100, 0.0009772
9: 0.0071583, 0.0108060, 0.0071063, 0.0107367, -0.0016366, 0.0017577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004283, upper bound: 0.0004340
time: 0.97 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004251
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041056, -0.0040758, -0.0000134, 0.0000143
1: -0.0063876, -0.0052785, -0.0064088, -0.0052944, -0.0005000, 0.0005370
2: 0.9687980, 0.9701290, 0.9687726, 0.9701099, -0.0006000, 0.0006444
3: 0.0161655, 0.0259825, 0.0159781, 0.0258420, -0.0044257, 0.0047530
4: -0.0026692, -0.0019225, -0.0026585, -0.0019083, -0.0003615, 0.0003366
5: 0.0145727, 0.0153273, 0.0145835, 0.0153417, -0.0003654, 0.0003402
6: 0.0044516, 0.0048186, 0.0044446, 0.0048134, -0.0001655, 0.0001777
7: -0.0145118, -0.0119677, -0.0144754, -0.0119191, -0.0012318, 0.0011470
8: 0.0052161, 0.0072346, 0.0052450, 0.0072731, -0.0009772, 0.0009100
9: 0.0071063, 0.0107367, 0.0071583, 0.0108060, -0.0017577, 0.0016366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004481, upper bound: 0.0004203
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004372, upper bound: 0.0004193
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041050, -0.0040754, -0.0000136, 0.0000136
1: -0.0063876, -0.0052785, -0.0063876, -0.0052785, -0.0005084, 0.0005084
2: 0.9687980, 0.9701290, 0.9687980, 0.9701290, -0.0006101, 0.0006101
3: 0.0161655, 0.0259825, 0.0161655, 0.0259825, -0.0044997, 0.0044997
4: -0.0026692, -0.0019225, -0.0026692, -0.0019225, -0.0003422, 0.0003422
5: 0.0145727, 0.0153273, 0.0145727, 0.0153273, -0.0003459, 0.0003459
6: 0.0044516, 0.0048186, 0.0044516, 0.0048186, -0.0001682, 0.0001682
7: -0.0145118, -0.0119677, -0.0145118, -0.0119677, -0.0011661, 0.0011661
8: 0.0052161, 0.0072346, 0.0052161, 0.0072346, -0.0009252, 0.0009252
9: 0.0071063, 0.0107367, 0.0071063, 0.0107367, -0.0016640, 0.0016640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004377, upper bound: 0.0004477
time: 0.93 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004372, upper bound: 0.0004459
time: 0.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.17 seconds
IS_A1_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004330
IS_A1_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
IS_A1_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004385
IS_A1_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004305
IS_A1_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004385, upper bound: 0.0004284
IS_A1_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
IS_A1_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004307, upper bound: 0.0004519
IS_A1_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004509
IS_A1_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004440, upper bound: 0.0004229
IS_A1_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
IS_A1_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004440, upper bound: 0.0004258
IS_A1_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004254
IS_A1_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004440, upper bound: 0.0004289
IS_A1_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004288
IS_A1_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004440, upper bound: 0.0004468
IS_A1_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004468
IS_A1_B1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004193, upper bound: 0.0004284
IS_A1_B1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
IS_A1_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004193, upper bound: 0.0004364
IS_A1_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004261
IS_A1_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004352, upper bound: 0.0004231
IS_A1_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
IS_A1_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004265, upper bound: 0.0004533
IS_A1_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004505
IS_A1_B1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004387, upper bound: 0.0004152
IS_A1_B1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
IS_A1_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004276, upper bound: 0.0004333
IS_A1_B1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004241
IS_A1_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004190
IS_A1_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
IS_A1_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004455
IS_A1_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004456
IS_A1_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004229, upper bound: 0.0004440
IS_A1_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
IS_A1_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004229, upper bound: 0.0004489
IS_A1_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004408
IS_A1_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004289, upper bound: 0.0004450
IS_A1_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
IS_A1_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004289, upper bound: 0.0004613
IS_A1_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004592
IS_A1_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004341, upper bound: 0.0004321
IS_A1_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
IS_A1_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004341, upper bound: 0.0004372
IS_A1_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004301
IS_A1_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004493, upper bound: 0.0004270
IS_A1_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
IS_A1_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004412, upper bound: 0.0004485
IS_A1_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004472
IS_A1_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004176, upper bound: 0.0004400
IS_A1_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
IS_A1_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004176, upper bound: 0.0004476
IS_A1_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004371
IS_A1_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004250, upper bound: 0.0004424
IS_A1_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
IS_A1_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004250, upper bound: 0.0004632
IS_A1_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004606
IS_A1_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004280, upper bound: 0.0004276
IS_A1_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
IS_A1_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004280, upper bound: 0.0004346
IS_A1_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004254
IS_A1_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004453, upper bound: 0.0004203
IS_A1_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
IS_A1_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004363, upper bound: 0.0004482
IS_A1_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004460
IS_A2_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004193
IS_A2_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004162
IS_A2_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004245
IS_A2_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004232
IS_A2_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004181, upper bound: 0.0004352
IS_A2_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004257
IS_A2_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004505
IS_A2_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004505
IS_A2_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004177
IS_A2_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004150
IS_A2_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004219
IS_A2_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004210
IS_A2_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004176
IS_A2_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004245
IS_A2_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004464
IS_A2_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004464
IS_A2_B1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004279
IS_A2_B1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004140, upper bound: 0.0004140
IS_A2_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004351
IS_A2_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004140, upper bound: 0.0004254
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004364, upper bound: 0.0004231
IS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004261, upper bound: 0.0004214
IS_A2_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004264, upper bound: 0.0004514
IS_A2_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004261, upper bound: 0.0004499
IS_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004152
IS_A2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004126
IS_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004190
IS_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004182
IS_A2_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004243
IS_A2_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004239
IS_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004455
IS_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004455
IS_A2_B2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004387
IS_A2_B2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004245
IS_A2_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004258, upper bound: 0.0004329
IS_A2_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004316
IS_A2_B2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004450
IS_A2_B2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004357
IS_A2_B2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004609
IS_A2_B2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004588
IS_A2_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004406, upper bound: 0.0004189
IS_A2_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004162
IS_A2_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004406, upper bound: 0.0004231
IS_A2_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004222
IS_A2_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004342
IS_A2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004257
IS_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004406, upper bound: 0.0004467
IS_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004467
IS_A2_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004387
IS_A2_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004126, upper bound: 0.0004243
IS_A2_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004450
IS_A2_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004126, upper bound: 0.0004357
IS_A2_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004246, upper bound: 0.0004413
IS_A2_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004241, upper bound: 0.0004311
IS_A2_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004246, upper bound: 0.0004609
IS_A2_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004241, upper bound: 0.0004587
IS_A2_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004283, upper bound: 0.0004275
IS_A2_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004142
IS_A2_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004283, upper bound: 0.0004340
IS_A2_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004251
IS_A2_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004481, upper bound: 0.0004203
IS_A2_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004372, upper bound: 0.0004193
IS_A2_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004377, upper bound: 0.0004477
IS_A2_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 2, lower bound: -0.0004372, upper bound: 0.0004459

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040766, -0.0041051, -0.0040765, -0.0000131, 0.0000130
1: -0.0063889, -0.0053259, -0.0063904, -0.0053218, -0.0004911, 0.0004882
2: 0.9687966, 0.9700723, 0.9687947, 0.9700771, -0.0005893, 0.0005859
3: 0.0161546, 0.0255636, 0.0161408, 0.0255996, -0.0043464, 0.0043213
4: -0.0026373, -0.0019217, -0.0026400, -0.0019206, -0.0003287, 0.0003306
5: 0.0146049, 0.0153281, 0.0146021, 0.0153292, -0.0003322, 0.0003341
6: 0.0044512, 0.0048030, 0.0044507, 0.0048043, -0.0001625, 0.0001616
7: -0.0144033, -0.0119649, -0.0144126, -0.0119613, -0.0011199, 0.0011264
8: 0.0053023, 0.0072368, 0.0052949, 0.0072396, -0.0008885, 0.0008936
9: 0.0072612, 0.0107407, 0.0072479, 0.0107458, -0.0015980, 0.0016073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040768, -0.0041050, -0.0040767, -0.0000134, 0.0000128
1: -0.0064025, -0.0053311, -0.0063895, -0.0053269, -0.0005029, 0.0004808
2: 0.9687803, 0.9700659, 0.9687957, 0.9700710, -0.0006034, 0.0005770
3: 0.0160343, 0.0255172, 0.0161486, 0.0255546, -0.0044509, 0.0042559
4: -0.0026338, -0.0019125, -0.0026366, -0.0019212, -0.0003237, 0.0003385
5: 0.0146085, 0.0153374, 0.0146056, 0.0153286, -0.0003271, 0.0003421
6: 0.0044467, 0.0048012, 0.0044510, 0.0048026, -0.0001664, 0.0001591
7: -0.0143912, -0.0119337, -0.0144009, -0.0119633, -0.0011029, 0.0011535
8: 0.0053118, 0.0072615, 0.0053041, 0.0072380, -0.0008750, 0.0009151
9: 0.0072784, 0.0107852, 0.0072646, 0.0107429, -0.0015738, 0.0016460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040766, -0.0041045, -0.0040761, -0.0000138, 0.0000128
1: -0.0063889, -0.0053259, -0.0063690, -0.0053059, -0.0005179, 0.0004786
2: 0.9687966, 0.9700723, 0.9688205, 0.9700961, -0.0006215, 0.0005743
3: 0.0161546, 0.0255636, 0.0163308, 0.0257404, -0.0045841, 0.0042359
4: -0.0026373, -0.0019217, -0.0026507, -0.0019351, -0.0003222, 0.0003486
5: 0.0146049, 0.0153281, 0.0145913, 0.0153146, -0.0003256, 0.0003524
6: 0.0044512, 0.0048030, 0.0044578, 0.0048096, -0.0001714, 0.0001584
7: -0.0144033, -0.0119649, -0.0144491, -0.0120105, -0.0010978, 0.0011880
8: 0.0053023, 0.0072368, 0.0052659, 0.0072006, -0.0008709, 0.0009425
9: 0.0072612, 0.0107407, 0.0071958, 0.0106755, -0.0015664, 0.0016952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004277, upper bound: 0.0004305
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004277, upper bound: 0.0004305
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040768, -0.0041045, -0.0040762, -0.0000142, 0.0000126
1: -0.0064025, -0.0053311, -0.0063681, -0.0053109, -0.0005299, 0.0004712
2: 0.9687803, 0.9700659, 0.9688215, 0.9700901, -0.0006360, 0.0005655
3: 0.0160343, 0.0255172, 0.0163385, 0.0256958, -0.0046907, 0.0041709
4: -0.0026338, -0.0019125, -0.0026473, -0.0019357, -0.0003172, 0.0003568
5: 0.0146085, 0.0153374, 0.0145947, 0.0153140, -0.0003206, 0.0003606
6: 0.0044467, 0.0048012, 0.0044581, 0.0048079, -0.0001754, 0.0001559
7: -0.0143912, -0.0119337, -0.0144375, -0.0120125, -0.0010809, 0.0012156
8: 0.0053118, 0.0072615, 0.0052751, 0.0071990, -0.0008576, 0.0009644
9: 0.0072784, 0.0107852, 0.0072124, 0.0106727, -0.0015424, 0.0017346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004277, upper bound: 0.0004305
time: 0.98 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004277, upper bound: 0.0004305
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041050, -0.0040766, -0.0000128, 0.0000138
1: -0.0063690, -0.0053059, -0.0063889, -0.0053259, -0.0004786, 0.0005179
2: 0.9688205, 0.9700961, 0.9687966, 0.9700723, -0.0005743, 0.0006215
3: 0.0163308, 0.0257404, 0.0161546, 0.0255636, -0.0042359, 0.0045841
4: -0.0026507, -0.0019351, -0.0026373, -0.0019217, -0.0003486, 0.0003222
5: 0.0145913, 0.0153146, 0.0146049, 0.0153281, -0.0003524, 0.0003256
6: 0.0044578, 0.0048096, 0.0044512, 0.0048030, -0.0001584, 0.0001714
7: -0.0144491, -0.0120105, -0.0144033, -0.0119649, -0.0011880, 0.0010978
8: 0.0052659, 0.0072006, 0.0053023, 0.0072368, -0.0009425, 0.0008709
9: 0.0071958, 0.0106755, 0.0072612, 0.0107407, -0.0016952, 0.0015664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040762, -0.0041054, -0.0040768, -0.0000126, 0.0000142
1: -0.0063681, -0.0053109, -0.0064025, -0.0053311, -0.0004712, 0.0005299
2: 0.9688215, 0.9700901, 0.9687803, 0.9700659, -0.0005655, 0.0006360
3: 0.0163385, 0.0256958, 0.0160343, 0.0255172, -0.0041709, 0.0046907
4: -0.0026473, -0.0019357, -0.0026338, -0.0019125, -0.0003568, 0.0003172
5: 0.0145947, 0.0153140, 0.0146085, 0.0153374, -0.0003606, 0.0003206
6: 0.0044581, 0.0048079, 0.0044467, 0.0048012, -0.0001559, 0.0001754
7: -0.0144375, -0.0120125, -0.0143912, -0.0119337, -0.0012156, 0.0010809
8: 0.0052751, 0.0071990, 0.0053118, 0.0072615, -0.0009644, 0.0008576
9: 0.0072124, 0.0106727, 0.0072784, 0.0107852, -0.0017346, 0.0015424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040762, -0.0041045, -0.0040761, -0.0000131, 0.0000130
1: -0.0063674, -0.0053099, -0.0063690, -0.0053059, -0.0004908, 0.0004880
2: 0.9688223, 0.9700913, 0.9688205, 0.9700961, -0.0005890, 0.0005856
3: 0.0163445, 0.0257047, 0.0163308, 0.0257404, -0.0043444, 0.0043193
4: -0.0026480, -0.0019361, -0.0026507, -0.0019351, -0.0003285, 0.0003304
5: 0.0145940, 0.0153135, 0.0145913, 0.0153146, -0.0003320, 0.0003339
6: 0.0044583, 0.0048082, 0.0044578, 0.0048096, -0.0001624, 0.0001615
7: -0.0144398, -0.0120141, -0.0144491, -0.0120105, -0.0011194, 0.0011259
8: 0.0052733, 0.0071978, 0.0052659, 0.0072006, -0.0008881, 0.0008932
9: 0.0072091, 0.0106705, 0.0071958, 0.0106755, -0.0015973, 0.0016065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004512, upper bound: 0.0004509
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004512, upper bound: 0.0004509
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040764, -0.0041045, -0.0040762, -0.0000134, 0.0000128
1: -0.0063815, -0.0053153, -0.0063681, -0.0053109, -0.0005027, 0.0004805
2: 0.9688054, 0.9700848, 0.9688215, 0.9700901, -0.0006033, 0.0005766
3: 0.0162200, 0.0256571, 0.0163385, 0.0256958, -0.0044495, 0.0042530
4: -0.0026444, -0.0019267, -0.0026473, -0.0019357, -0.0003235, 0.0003384
5: 0.0145977, 0.0153231, 0.0145947, 0.0153140, -0.0003269, 0.0003420
6: 0.0044536, 0.0048065, 0.0044581, 0.0048079, -0.0001664, 0.0001590
7: -0.0144275, -0.0119818, -0.0144375, -0.0120125, -0.0011022, 0.0011531
8: 0.0052830, 0.0072233, 0.0052751, 0.0071990, -0.0008744, 0.0009148
9: 0.0072267, 0.0107165, 0.0072124, 0.0106727, -0.0015727, 0.0016454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004512, upper bound: 0.0004509
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004512, upper bound: 0.0004509
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041050, -0.0040766, -0.0000138, 0.0000139
1: -0.0064082, -0.0053117, -0.0063889, -0.0053259, -0.0005164, 0.0005219
2: 0.9687734, 0.9700892, 0.9687966, 0.9700723, -0.0006198, 0.0006264
3: 0.0159837, 0.0256892, 0.0161546, 0.0255636, -0.0045713, 0.0046199
4: -0.0026468, -0.0019087, -0.0026373, -0.0019217, -0.0003514, 0.0003477
5: 0.0145952, 0.0153413, 0.0146049, 0.0153281, -0.0003551, 0.0003514
6: 0.0044448, 0.0048077, 0.0044512, 0.0048030, -0.0001709, 0.0001727
7: -0.0144358, -0.0119206, -0.0144033, -0.0119649, -0.0011973, 0.0011847
8: 0.0052764, 0.0072719, 0.0053023, 0.0072368, -0.0009499, 0.0009399
9: 0.0072148, 0.0108039, 0.0072612, 0.0107407, -0.0017084, 0.0016904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040764, -0.0041054, -0.0040768, -0.0000136, 0.0000143
1: -0.0064072, -0.0053168, -0.0064025, -0.0053311, -0.0005089, 0.0005337
2: 0.9687745, 0.9700830, 0.9687803, 0.9700659, -0.0006107, 0.0006404
3: 0.0159921, 0.0256434, 0.0160343, 0.0255172, -0.0045043, 0.0047238
4: -0.0026434, -0.0019093, -0.0026338, -0.0019125, -0.0003593, 0.0003426
5: 0.0145988, 0.0153406, 0.0146085, 0.0153374, -0.0003631, 0.0003462
6: 0.0044451, 0.0048060, 0.0044467, 0.0048012, -0.0001684, 0.0001766
7: -0.0144240, -0.0119227, -0.0143912, -0.0119337, -0.0012242, 0.0011673
8: 0.0052858, 0.0072702, 0.0053118, 0.0072615, -0.0009712, 0.0009261
9: 0.0072317, 0.0108008, 0.0072784, 0.0107852, -0.0017469, 0.0016657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041050, -0.0040766, -0.0000135, 0.0000146
1: -0.0063873, -0.0052954, -0.0063889, -0.0053259, -0.0005056, 0.0005470
2: 0.9687985, 0.9701087, 0.9687966, 0.9700723, -0.0006068, 0.0006565
3: 0.0161682, 0.0258329, 0.0161546, 0.0255636, -0.0044755, 0.0048420
4: -0.0026578, -0.0019227, -0.0026373, -0.0019217, -0.0003683, 0.0003404
5: 0.0145842, 0.0153271, 0.0146049, 0.0153281, -0.0003722, 0.0003440
6: 0.0044517, 0.0048130, 0.0044512, 0.0048030, -0.0001673, 0.0001810
7: -0.0144731, -0.0119684, -0.0144033, -0.0119649, -0.0012549, 0.0011599
8: 0.0052469, 0.0072340, 0.0053023, 0.0072368, -0.0009955, 0.0009202
9: 0.0071617, 0.0107357, 0.0072612, 0.0107407, -0.0017906, 0.0016550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004408, upper bound: 0.0004254
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004408, upper bound: 0.0004254
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040760, -0.0041054, -0.0040768, -0.0000133, 0.0000149
1: -0.0063864, -0.0053004, -0.0064025, -0.0053311, -0.0004981, 0.0005587
2: 0.9687995, 0.9701028, 0.9687803, 0.9700659, -0.0005977, 0.0006705
3: 0.0161765, 0.0257885, 0.0160343, 0.0255172, -0.0044088, 0.0049452
4: -0.0026544, -0.0019233, -0.0026338, -0.0019125, -0.0003761, 0.0003353
5: 0.0145876, 0.0153265, 0.0146085, 0.0153374, -0.0003801, 0.0003389
6: 0.0044520, 0.0048114, 0.0044467, 0.0048012, -0.0001648, 0.0001849
7: -0.0144616, -0.0119705, -0.0143912, -0.0119337, -0.0012816, 0.0011426
8: 0.0052560, 0.0072323, 0.0053118, 0.0072615, -0.0010168, 0.0009065
9: 0.0071781, 0.0107326, 0.0072784, 0.0107852, -0.0018287, 0.0016304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004408, upper bound: 0.0004254
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004408, upper bound: 0.0004254
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041044, -0.0040762, -0.0000145, 0.0000137
1: -0.0064082, -0.0053117, -0.0063674, -0.0053099, -0.0005434, 0.0005124
2: 0.9687734, 0.9700892, 0.9688223, 0.9700913, -0.0006521, 0.0006149
3: 0.0159837, 0.0256892, 0.0163445, 0.0257047, -0.0048101, 0.0045352
4: -0.0026468, -0.0019087, -0.0026480, -0.0019361, -0.0003449, 0.0003658
5: 0.0145952, 0.0153413, 0.0145940, 0.0153135, -0.0003486, 0.0003697
6: 0.0044448, 0.0048077, 0.0044583, 0.0048082, -0.0001798, 0.0001696
7: -0.0144358, -0.0119206, -0.0144398, -0.0120141, -0.0011753, 0.0012466
8: 0.0052764, 0.0072719, 0.0052733, 0.0071978, -0.0009325, 0.0009890
9: 0.0072148, 0.0108039, 0.0072091, 0.0106705, -0.0016771, 0.0017788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004288
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004288
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040764, -0.0041048, -0.0040764, -0.0000143, 0.0000140
1: -0.0064072, -0.0053168, -0.0063815, -0.0053153, -0.0005364, 0.0005235
2: 0.9687745, 0.9700830, 0.9688054, 0.9700848, -0.0006437, 0.0006282
3: 0.0159921, 0.0256434, 0.0162200, 0.0256571, -0.0047476, 0.0046337
4: -0.0026434, -0.0019093, -0.0026444, -0.0019267, -0.0003524, 0.0003611
5: 0.0145988, 0.0153406, 0.0145977, 0.0153231, -0.0003562, 0.0003649
6: 0.0044451, 0.0048060, 0.0044536, 0.0048065, -0.0001775, 0.0001732
7: -0.0144240, -0.0119227, -0.0144275, -0.0119818, -0.0012009, 0.0012304
8: 0.0052858, 0.0072702, 0.0052830, 0.0072233, -0.0009527, 0.0009761
9: 0.0072317, 0.0108008, 0.0072267, 0.0107165, -0.0017135, 0.0017557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004288
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004288
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041044, -0.0040762, -0.0000138, 0.0000139
1: -0.0063873, -0.0052954, -0.0063674, -0.0053099, -0.0005165, 0.0005217
2: 0.9687985, 0.9701087, 0.9688223, 0.9700913, -0.0006198, 0.0006261
3: 0.0161682, 0.0258329, 0.0163445, 0.0257047, -0.0045713, 0.0046180
4: -0.0026578, -0.0019227, -0.0026480, -0.0019361, -0.0003512, 0.0003477
5: 0.0145842, 0.0153271, 0.0145940, 0.0153135, -0.0003550, 0.0003514
6: 0.0044517, 0.0048130, 0.0044583, 0.0048082, -0.0001709, 0.0001727
7: -0.0144731, -0.0119684, -0.0144398, -0.0120141, -0.0011968, 0.0011847
8: 0.0052469, 0.0072340, 0.0052733, 0.0071978, -0.0009495, 0.0009399
9: 0.0071617, 0.0107357, 0.0072091, 0.0106705, -0.0017077, 0.0016905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004590, upper bound: 0.0004468
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004590, upper bound: 0.0004468
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040760, -0.0041048, -0.0040764, -0.0000136, 0.0000142
1: -0.0063864, -0.0053004, -0.0063815, -0.0053153, -0.0005088, 0.0005336
2: 0.9687995, 0.9701028, 0.9688054, 0.9700848, -0.0006106, 0.0006403
3: 0.0161765, 0.0257885, 0.0162200, 0.0256571, -0.0045037, 0.0047228
4: -0.0026544, -0.0019233, -0.0026444, -0.0019267, -0.0003592, 0.0003425
5: 0.0145876, 0.0153265, 0.0145977, 0.0153231, -0.0003630, 0.0003462
6: 0.0044520, 0.0048114, 0.0044536, 0.0048065, -0.0001684, 0.0001766
7: -0.0144616, -0.0119705, -0.0144275, -0.0119818, -0.0012240, 0.0011672
8: 0.0052560, 0.0072323, 0.0052830, 0.0072233, -0.0009710, 0.0009260
9: 0.0071781, 0.0107326, 0.0072267, 0.0107165, -0.0017465, 0.0016655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004590, upper bound: 0.0004468
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004590, upper bound: 0.0004468
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040766, -0.0041050, -0.0040761, -0.0000138, 0.0000133
1: -0.0063889, -0.0053259, -0.0063893, -0.0053056, -0.0005171, 0.0004976
2: 0.9687966, 0.9700723, 0.9687960, 0.9700965, -0.0006206, 0.0005972
3: 0.0161546, 0.0255636, 0.0161506, 0.0257431, -0.0045774, 0.0044048
4: -0.0026373, -0.0019217, -0.0026509, -0.0019214, -0.0003350, 0.0003481
5: 0.0146049, 0.0153281, 0.0145911, 0.0153285, -0.0003386, 0.0003519
6: 0.0044512, 0.0048030, 0.0044510, 0.0048097, -0.0001711, 0.0001647
7: -0.0144033, -0.0119649, -0.0144498, -0.0119638, -0.0011416, 0.0011863
8: 0.0053023, 0.0072368, 0.0052654, 0.0072376, -0.0009057, 0.0009411
9: 0.0072612, 0.0107407, 0.0071949, 0.0107422, -0.0016289, 0.0016927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040768, -0.0041050, -0.0040762, -0.0000141, 0.0000131
1: -0.0064025, -0.0053311, -0.0063884, -0.0053109, -0.0005288, 0.0004902
2: 0.9687803, 0.9700659, 0.9687971, 0.9700902, -0.0006346, 0.0005883
3: 0.0160343, 0.0255172, 0.0161585, 0.0256961, -0.0046807, 0.0043392
4: -0.0026338, -0.0019125, -0.0026474, -0.0019220, -0.0003300, 0.0003560
5: 0.0146085, 0.0153374, 0.0145947, 0.0153278, -0.0003335, 0.0003598
6: 0.0044467, 0.0048012, 0.0044513, 0.0048079, -0.0001750, 0.0001622
7: -0.0143912, -0.0119337, -0.0144376, -0.0119659, -0.0011246, 0.0012130
8: 0.0053118, 0.0072615, 0.0052750, 0.0072360, -0.0008922, 0.0009624
9: 0.0072784, 0.0107852, 0.0072122, 0.0107393, -0.0016046, 0.0017309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040766, -0.0041044, -0.0040757, -0.0000144, 0.0000129
1: -0.0063889, -0.0053259, -0.0063675, -0.0052901, -0.0005389, 0.0004821
2: 0.9687966, 0.9700723, 0.9688222, 0.9701151, -0.0006467, 0.0005785
3: 0.0161546, 0.0255636, 0.0163433, 0.0258797, -0.0047699, 0.0042672
4: -0.0026373, -0.0019217, -0.0026613, -0.0019360, -0.0003245, 0.0003628
5: 0.0146049, 0.0153281, 0.0145806, 0.0153136, -0.0003280, 0.0003667
6: 0.0044512, 0.0048030, 0.0044582, 0.0048148, -0.0001783, 0.0001595
7: -0.0144033, -0.0119649, -0.0144852, -0.0120138, -0.0011059, 0.0012362
8: 0.0053023, 0.0072368, 0.0052373, 0.0071980, -0.0008773, 0.0009807
9: 0.0072612, 0.0107407, 0.0071444, 0.0106709, -0.0015780, 0.0017639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004232, upper bound: 0.0004261
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004232, upper bound: 0.0004261
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040768, -0.0041044, -0.0040758, -0.0000147, 0.0000127
1: -0.0064025, -0.0053311, -0.0063666, -0.0052954, -0.0005507, 0.0004747
2: 0.9687803, 0.9700659, 0.9688233, 0.9701087, -0.0006608, 0.0005696
3: 0.0160343, 0.0255172, 0.0163515, 0.0258332, -0.0048742, 0.0042013
4: -0.0026338, -0.0019125, -0.0026578, -0.0019367, -0.0003195, 0.0003707
5: 0.0146085, 0.0153374, 0.0145842, 0.0153130, -0.0003229, 0.0003747
6: 0.0044467, 0.0048012, 0.0044585, 0.0048130, -0.0001822, 0.0001571
7: -0.0143912, -0.0119337, -0.0144732, -0.0120159, -0.0010888, 0.0012632
8: 0.0053118, 0.0072615, 0.0052468, 0.0071963, -0.0008638, 0.0010022
9: 0.0072784, 0.0107852, 0.0071615, 0.0106679, -0.0015536, 0.0018025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004232, upper bound: 0.0004261
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004232, upper bound: 0.0004261
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040761, -0.0041050, -0.0040762, -0.0000135, 0.0000141
1: -0.0063690, -0.0053059, -0.0063878, -0.0053094, -0.0005050, 0.0005274
2: 0.9688205, 0.9700961, 0.9687978, 0.9700920, -0.0006060, 0.0006329
3: 0.0163308, 0.0257404, 0.0161639, 0.0257091, -0.0044697, 0.0046679
4: -0.0026507, -0.0019351, -0.0026484, -0.0019224, -0.0003550, 0.0003399
5: 0.0145913, 0.0153146, 0.0145937, 0.0153274, -0.0003588, 0.0003436
6: 0.0044578, 0.0048096, 0.0044515, 0.0048084, -0.0001671, 0.0001745
7: -0.0144491, -0.0120105, -0.0144410, -0.0119673, -0.0012097, 0.0011584
8: 0.0052659, 0.0072006, 0.0052723, 0.0072349, -0.0009597, 0.0009190
9: 0.0071958, 0.0106755, 0.0072074, 0.0107372, -0.0017262, 0.0016529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040762, -0.0041053, -0.0040764, -0.0000133, 0.0000143
1: -0.0063681, -0.0053109, -0.0064001, -0.0053155, -0.0004981, 0.0005369
2: 0.9688215, 0.9700901, 0.9687831, 0.9700847, -0.0005978, 0.0006443
3: 0.0163385, 0.0256958, 0.0160549, 0.0256556, -0.0044093, 0.0047523
4: -0.0026473, -0.0019357, -0.0026443, -0.0019141, -0.0003614, 0.0003354
5: 0.0145947, 0.0153140, 0.0145978, 0.0153358, -0.0003653, 0.0003389
6: 0.0044581, 0.0048079, 0.0044475, 0.0048064, -0.0001649, 0.0001777
7: -0.0144375, -0.0120125, -0.0144271, -0.0119390, -0.0012316, 0.0011427
8: 0.0052751, 0.0071990, 0.0052833, 0.0072573, -0.0009771, 0.0009066
9: 0.0072124, 0.0106727, 0.0072272, 0.0107775, -0.0017574, 0.0016305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040762, -0.0041044, -0.0040757, -0.0000138, 0.0000133
1: -0.0063674, -0.0053099, -0.0063675, -0.0052901, -0.0005173, 0.0004976
2: 0.9688223, 0.9700913, 0.9688222, 0.9701151, -0.0006208, 0.0005971
3: 0.0163445, 0.0257047, 0.0163433, 0.0258797, -0.0045789, 0.0044041
4: -0.0026480, -0.0019361, -0.0026613, -0.0019360, -0.0003350, 0.0003483
5: 0.0145940, 0.0153135, 0.0145806, 0.0153136, -0.0003385, 0.0003520
6: 0.0044583, 0.0048082, 0.0044582, 0.0048148, -0.0001712, 0.0001647
7: -0.0144398, -0.0120141, -0.0144852, -0.0120138, -0.0011414, 0.0011867
8: 0.0052733, 0.0071978, 0.0052373, 0.0071980, -0.0009055, 0.0009414
9: 0.0072091, 0.0106705, 0.0071444, 0.0106709, -0.0016286, 0.0016933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004504
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004505
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040764, -0.0041044, -0.0040758, -0.0000141, 0.0000131
1: -0.0063815, -0.0053153, -0.0063666, -0.0052954, -0.0005291, 0.0004900
2: 0.9688054, 0.9700848, 0.9688233, 0.9701087, -0.0006349, 0.0005880
3: 0.0162200, 0.0256571, 0.0163515, 0.0258332, -0.0046828, 0.0043373
4: -0.0026444, -0.0019267, -0.0026578, -0.0019367, -0.0003299, 0.0003562
5: 0.0145977, 0.0153231, 0.0145842, 0.0153130, -0.0003334, 0.0003600
6: 0.0044536, 0.0048065, 0.0044585, 0.0048130, -0.0001751, 0.0001622
7: -0.0144275, -0.0119818, -0.0144732, -0.0120159, -0.0011241, 0.0012136
8: 0.0052830, 0.0072233, 0.0052468, 0.0071963, -0.0008918, 0.0009628
9: 0.0072267, 0.0107165, 0.0071615, 0.0106679, -0.0016039, 0.0017317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004504
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004505
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040763, -0.0041050, -0.0040762, -0.0000145, 0.0000142
1: -0.0064082, -0.0053117, -0.0063878, -0.0053094, -0.0005429, 0.0005314
2: 0.9687734, 0.9700892, 0.9687978, 0.9700920, -0.0006515, 0.0006377
3: 0.0159837, 0.0256892, 0.0161639, 0.0257091, -0.0048051, 0.0047036
4: -0.0026468, -0.0019087, -0.0026484, -0.0019224, -0.0003577, 0.0003655
5: 0.0145952, 0.0153413, 0.0145937, 0.0153274, -0.0003616, 0.0003694
6: 0.0044448, 0.0048077, 0.0044515, 0.0048084, -0.0001797, 0.0001759
7: -0.0144358, -0.0119206, -0.0144410, -0.0119673, -0.0012190, 0.0012453
8: 0.0052764, 0.0072719, 0.0052723, 0.0072349, -0.0009671, 0.0009880
9: 0.0072148, 0.0108039, 0.0072074, 0.0107372, -0.0017394, 0.0017769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040764, -0.0041053, -0.0040764, -0.0000143, 0.0000144
1: -0.0064072, -0.0053168, -0.0064001, -0.0053155, -0.0005358, 0.0005406
2: 0.9687745, 0.9700830, 0.9687831, 0.9700847, -0.0006430, 0.0006488
3: 0.0159921, 0.0256434, 0.0160549, 0.0256556, -0.0047427, 0.0047854
4: -0.0026434, -0.0019093, -0.0026443, -0.0019141, -0.0003640, 0.0003607
5: 0.0145988, 0.0153406, 0.0145978, 0.0153358, -0.0003678, 0.0003646
6: 0.0044451, 0.0048060, 0.0044475, 0.0048064, -0.0001773, 0.0001789
7: -0.0144240, -0.0119227, -0.0144271, -0.0119390, -0.0012402, 0.0012291
8: 0.0052858, 0.0072702, 0.0052833, 0.0072573, -0.0009839, 0.0009751
9: 0.0072317, 0.0108008, 0.0072272, 0.0107775, -0.0017696, 0.0017538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040764, -0.0041044, -0.0040757, -0.0000151, 0.0000137
1: -0.0064066, -0.0053156, -0.0063675, -0.0052901, -0.0005670, 0.0005131
2: 0.9687753, 0.9700845, 0.9688222, 0.9701151, -0.0006804, 0.0006158
3: 0.0159975, 0.0256543, 0.0163433, 0.0258797, -0.0050184, 0.0045419
4: -0.0026442, -0.0019097, -0.0026613, -0.0019360, -0.0003454, 0.0003817
5: 0.0145979, 0.0153402, 0.0145806, 0.0153136, -0.0003491, 0.0003858
6: 0.0044453, 0.0048064, 0.0044582, 0.0048148, -0.0001876, 0.0001698
7: -0.0144268, -0.0119241, -0.0144852, -0.0120138, -0.0011771, 0.0013006
8: 0.0052836, 0.0072691, 0.0052373, 0.0071980, -0.0009338, 0.0010318
9: 0.0072277, 0.0107988, 0.0071444, 0.0106709, -0.0016796, 0.0018558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004316, upper bound: 0.0004241
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004316, upper bound: 0.0004241
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040765, -0.0041044, -0.0040758, -0.0000154, 0.0000135
1: -0.0064187, -0.0053217, -0.0063666, -0.0052954, -0.0005768, 0.0005055
2: 0.9687607, 0.9700772, 0.9688233, 0.9701087, -0.0006922, 0.0006066
3: 0.0158907, 0.0256007, 0.0163515, 0.0258332, -0.0051052, 0.0044739
4: -0.0026401, -0.0019016, -0.0026578, -0.0019367, -0.0003403, 0.0003883
5: 0.0146020, 0.0153484, 0.0145842, 0.0153130, -0.0003439, 0.0003924
6: 0.0044413, 0.0048044, 0.0044585, 0.0048130, -0.0001909, 0.0001673
7: -0.0144129, -0.0118965, -0.0144732, -0.0120159, -0.0011595, 0.0013231
8: 0.0052946, 0.0072910, 0.0052468, 0.0071963, -0.0009199, 0.0010496
9: 0.0072475, 0.0108383, 0.0071615, 0.0106679, -0.0016545, 0.0018879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004316, upper bound: 0.0004241
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004316, upper bound: 0.0004241
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041050, -0.0040762, -0.0000142, 0.0000149
1: -0.0063873, -0.0052954, -0.0063878, -0.0053094, -0.0005321, 0.0005565
2: 0.9687985, 0.9701087, 0.9687978, 0.9700920, -0.0006385, 0.0006678
3: 0.0161682, 0.0258329, 0.0161639, 0.0257091, -0.0047094, 0.0049258
4: -0.0026578, -0.0019227, -0.0026484, -0.0019224, -0.0003746, 0.0003582
5: 0.0145842, 0.0153271, 0.0145937, 0.0153274, -0.0003786, 0.0003620
6: 0.0044517, 0.0048130, 0.0044515, 0.0048084, -0.0001761, 0.0001842
7: -0.0144731, -0.0119684, -0.0144410, -0.0119673, -0.0012766, 0.0012205
8: 0.0052469, 0.0072340, 0.0052723, 0.0072349, -0.0010128, 0.0009683
9: 0.0071617, 0.0107357, 0.0072074, 0.0107372, -0.0018215, 0.0017415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040760, -0.0041053, -0.0040764, -0.0000140, 0.0000151
1: -0.0063864, -0.0053004, -0.0064001, -0.0053155, -0.0005250, 0.0005657
2: 0.9687995, 0.9701028, 0.9687831, 0.9700847, -0.0006301, 0.0006788
3: 0.0161765, 0.0257885, 0.0160549, 0.0256556, -0.0046472, 0.0050068
4: -0.0026544, -0.0019233, -0.0026443, -0.0019141, -0.0003808, 0.0003534
5: 0.0145876, 0.0153265, 0.0145978, 0.0153358, -0.0003849, 0.0003572
6: 0.0044520, 0.0048114, 0.0044475, 0.0048064, -0.0001738, 0.0001872
7: -0.0144616, -0.0119705, -0.0144271, -0.0119390, -0.0012975, 0.0012044
8: 0.0052560, 0.0072323, 0.0052833, 0.0072573, -0.0010294, 0.0009555
9: 0.0071781, 0.0107326, 0.0072272, 0.0107775, -0.0018515, 0.0017185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041044, -0.0040758, -0.0000145, 0.0000142
1: -0.0063873, -0.0052954, -0.0063660, -0.0052940, -0.0005433, 0.0005313
2: 0.9687985, 0.9701087, 0.9688240, 0.9701105, -0.0006520, 0.0006376
3: 0.0161682, 0.0258329, 0.0163571, 0.0258457, -0.0048090, 0.0047026
4: -0.0026578, -0.0019227, -0.0026587, -0.0019371, -0.0003577, 0.0003658
5: 0.0145842, 0.0153271, 0.0145832, 0.0153126, -0.0003615, 0.0003697
6: 0.0044517, 0.0048130, 0.0044587, 0.0048135, -0.0001798, 0.0001758
7: -0.0144731, -0.0119684, -0.0144764, -0.0120173, -0.0012187, 0.0012463
8: 0.0052469, 0.0072340, 0.0052443, 0.0071952, -0.0009669, 0.0009887
9: 0.0071617, 0.0107357, 0.0071569, 0.0106658, -0.0017390, 0.0017784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004585, upper bound: 0.0004455
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004585, upper bound: 0.0004455
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040760, -0.0041047, -0.0040759, -0.0000143, 0.0000144
1: -0.0063864, -0.0053004, -0.0063779, -0.0053000, -0.0005363, 0.0005407
2: 0.9687995, 0.9701028, 0.9688098, 0.9701032, -0.0006435, 0.0006488
3: 0.0161765, 0.0257885, 0.0162520, 0.0257923, -0.0047467, 0.0047856
4: -0.0026544, -0.0019233, -0.0026547, -0.0019291, -0.0003640, 0.0003610
5: 0.0145876, 0.0153265, 0.0145873, 0.0153207, -0.0003679, 0.0003649
6: 0.0044520, 0.0048114, 0.0044548, 0.0048115, -0.0001775, 0.0001789
7: -0.0144616, -0.0119705, -0.0144626, -0.0119901, -0.0012402, 0.0012301
8: 0.0052560, 0.0072323, 0.0052552, 0.0072168, -0.0009839, 0.0009759
9: 0.0071781, 0.0107326, 0.0071767, 0.0107047, -0.0017697, 0.0017553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004585, upper bound: 0.0004456
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004585, upper bound: 0.0004456
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040766, -0.0041055, -0.0040763, -0.0000139, 0.0000138
1: -0.0063889, -0.0053259, -0.0064082, -0.0053117, -0.0005219, 0.0005164
2: 0.9687966, 0.9700723, 0.9687734, 0.9700892, -0.0006264, 0.0006198
3: 0.0161546, 0.0255636, 0.0159837, 0.0256892, -0.0046199, 0.0045713
4: -0.0026373, -0.0019217, -0.0026468, -0.0019087, -0.0003477, 0.0003514
5: 0.0146049, 0.0153281, 0.0145952, 0.0153413, -0.0003514, 0.0003551
6: 0.0044512, 0.0048030, 0.0044448, 0.0048077, -0.0001727, 0.0001709
7: -0.0144033, -0.0119649, -0.0144358, -0.0119206, -0.0011847, 0.0011973
8: 0.0053023, 0.0072368, 0.0052764, 0.0072719, -0.0009399, 0.0009499
9: 0.0072612, 0.0107407, 0.0072148, 0.0108039, -0.0016904, 0.0017084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040768, -0.0041055, -0.0040764, -0.0000143, 0.0000136
1: -0.0064025, -0.0053311, -0.0064072, -0.0053168, -0.0005337, 0.0005089
2: 0.9687803, 0.9700659, 0.9687745, 0.9700830, -0.0006404, 0.0006107
3: 0.0160343, 0.0255172, 0.0159921, 0.0256434, -0.0047238, 0.0045043
4: -0.0026338, -0.0019125, -0.0026434, -0.0019093, -0.0003426, 0.0003593
5: 0.0146085, 0.0153374, 0.0145988, 0.0153406, -0.0003462, 0.0003631
6: 0.0044467, 0.0048012, 0.0044451, 0.0048060, -0.0001766, 0.0001684
7: -0.0143912, -0.0119337, -0.0144240, -0.0119227, -0.0011673, 0.0012242
8: 0.0053118, 0.0072615, 0.0052858, 0.0072702, -0.0009261, 0.0009712
9: 0.0072784, 0.0107852, 0.0072317, 0.0108008, -0.0016657, 0.0017469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
time: 0.91 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040766, -0.0041050, -0.0040758, -0.0000146, 0.0000135
1: -0.0063889, -0.0053259, -0.0063873, -0.0052954, -0.0005470, 0.0005056
2: 0.9687966, 0.9700723, 0.9687985, 0.9701087, -0.0006565, 0.0006068
3: 0.0161546, 0.0255636, 0.0161682, 0.0258329, -0.0048420, 0.0044755
4: -0.0026373, -0.0019217, -0.0026578, -0.0019227, -0.0003404, 0.0003683
5: 0.0146049, 0.0153281, 0.0145842, 0.0153271, -0.0003440, 0.0003722
6: 0.0044512, 0.0048030, 0.0044517, 0.0048130, -0.0001810, 0.0001673
7: -0.0144033, -0.0119649, -0.0144731, -0.0119684, -0.0011599, 0.0012549
8: 0.0053023, 0.0072368, 0.0052469, 0.0072340, -0.0009202, 0.0009955
9: 0.0072612, 0.0107407, 0.0071617, 0.0107357, -0.0016550, 0.0017906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004408
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004408
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040768, -0.0041050, -0.0040760, -0.0000149, 0.0000133
1: -0.0064025, -0.0053311, -0.0063864, -0.0053004, -0.0005587, 0.0004981
2: 0.9687803, 0.9700659, 0.9687995, 0.9701028, -0.0006705, 0.0005977
3: 0.0160343, 0.0255172, 0.0161765, 0.0257885, -0.0049452, 0.0044088
4: -0.0026338, -0.0019125, -0.0026544, -0.0019233, -0.0003353, 0.0003761
5: 0.0146085, 0.0153374, 0.0145876, 0.0153265, -0.0003389, 0.0003801
6: 0.0044467, 0.0048012, 0.0044520, 0.0048114, -0.0001849, 0.0001648
7: -0.0143912, -0.0119337, -0.0144616, -0.0119705, -0.0011426, 0.0012816
8: 0.0053118, 0.0072615, 0.0052560, 0.0072323, -0.0009065, 0.0010168
9: 0.0072784, 0.0107852, 0.0071781, 0.0107326, -0.0016304, 0.0018287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004408
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004408
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040762, -0.0041055, -0.0040763, -0.0000137, 0.0000145
1: -0.0063674, -0.0053099, -0.0064082, -0.0053117, -0.0005124, 0.0005434
2: 0.9688223, 0.9700913, 0.9687734, 0.9700892, -0.0006149, 0.0006521
3: 0.0163445, 0.0257047, 0.0159837, 0.0256892, -0.0045352, 0.0048101
4: -0.0026480, -0.0019361, -0.0026468, -0.0019087, -0.0003658, 0.0003449
5: 0.0145940, 0.0153135, 0.0145952, 0.0153413, -0.0003697, 0.0003486
6: 0.0044583, 0.0048082, 0.0044448, 0.0048077, -0.0001696, 0.0001798
7: -0.0144398, -0.0120141, -0.0144358, -0.0119206, -0.0012466, 0.0011753
8: 0.0052733, 0.0071978, 0.0052764, 0.0072719, -0.0009890, 0.0009325
9: 0.0072091, 0.0106705, 0.0072148, 0.0108039, -0.0017788, 0.0016771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
time: 0.84 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040764, -0.0041055, -0.0040764, -0.0000140, 0.0000143
1: -0.0063815, -0.0053153, -0.0064072, -0.0053168, -0.0005235, 0.0005364
2: 0.9688054, 0.9700848, 0.9687745, 0.9700830, -0.0006282, 0.0006437
3: 0.0162200, 0.0256571, 0.0159921, 0.0256434, -0.0046337, 0.0047476
4: -0.0026444, -0.0019267, -0.0026434, -0.0019093, -0.0003611, 0.0003524
5: 0.0145977, 0.0153231, 0.0145988, 0.0153406, -0.0003649, 0.0003562
6: 0.0044536, 0.0048065, 0.0044451, 0.0048060, -0.0001732, 0.0001775
7: -0.0144275, -0.0119818, -0.0144240, -0.0119227, -0.0012304, 0.0012009
8: 0.0052830, 0.0072233, 0.0052858, 0.0072702, -0.0009761, 0.0009527
9: 0.0072267, 0.0107165, 0.0072317, 0.0108008, -0.0017557, 0.0017135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040762, -0.0041050, -0.0040758, -0.0000139, 0.0000138
1: -0.0063674, -0.0053099, -0.0063873, -0.0052954, -0.0005217, 0.0005165
2: 0.9688223, 0.9700913, 0.9687985, 0.9701087, -0.0006261, 0.0006198
3: 0.0163445, 0.0257047, 0.0161682, 0.0258329, -0.0046180, 0.0045713
4: -0.0026480, -0.0019361, -0.0026578, -0.0019227, -0.0003477, 0.0003512
5: 0.0145940, 0.0153135, 0.0145842, 0.0153271, -0.0003514, 0.0003550
6: 0.0044583, 0.0048082, 0.0044517, 0.0048130, -0.0001727, 0.0001709
7: -0.0144398, -0.0120141, -0.0144731, -0.0119684, -0.0011847, 0.0011968
8: 0.0052733, 0.0071978, 0.0052469, 0.0072340, -0.0009399, 0.0009495
9: 0.0072091, 0.0106705, 0.0071617, 0.0107357, -0.0016905, 0.0017077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004588
time: 0.91 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004592
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040764, -0.0041050, -0.0040760, -0.0000142, 0.0000136
1: -0.0063815, -0.0053153, -0.0063864, -0.0053004, -0.0005336, 0.0005088
2: 0.9688054, 0.9700848, 0.9687995, 0.9701028, -0.0006403, 0.0006106
3: 0.0162200, 0.0256571, 0.0161765, 0.0257885, -0.0047228, 0.0045037
4: -0.0026444, -0.0019267, -0.0026544, -0.0019233, -0.0003425, 0.0003592
5: 0.0145977, 0.0153231, 0.0145876, 0.0153265, -0.0003462, 0.0003630
6: 0.0044536, 0.0048065, 0.0044520, 0.0048114, -0.0001766, 0.0001684
7: -0.0144275, -0.0119818, -0.0144616, -0.0119705, -0.0011672, 0.0012240
8: 0.0052830, 0.0072233, 0.0052560, 0.0072323, -0.0009260, 0.0009710
9: 0.0072267, 0.0107165, 0.0071781, 0.0107326, -0.0016655, 0.0017465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004588
time: 0.87 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004592
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040764, -0.0041055, -0.0040763, -0.0000135, 0.0000134
1: -0.0064066, -0.0053156, -0.0064082, -0.0053117, -0.0005043, 0.0005015
2: 0.9687753, 0.9700845, 0.9687734, 0.9700892, -0.0006052, 0.0006018
3: 0.0159975, 0.0256543, 0.0159837, 0.0256892, -0.0044641, 0.0044389
4: -0.0026442, -0.0019097, -0.0026468, -0.0019087, -0.0003376, 0.0003395
5: 0.0145979, 0.0153402, 0.0145952, 0.0153413, -0.0003412, 0.0003431
6: 0.0044453, 0.0048064, 0.0044448, 0.0048077, -0.0001669, 0.0001660
7: -0.0144268, -0.0119241, -0.0144358, -0.0119206, -0.0011504, 0.0011569
8: 0.0052836, 0.0072691, 0.0052764, 0.0072719, -0.0009127, 0.0009178
9: 0.0072277, 0.0107988, 0.0072148, 0.0108039, -0.0016415, 0.0016508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040765, -0.0041055, -0.0040764, -0.0000138, 0.0000132
1: -0.0064187, -0.0053217, -0.0064072, -0.0053168, -0.0005156, 0.0004941
2: 0.9687607, 0.9700772, 0.9687745, 0.9700830, -0.0006187, 0.0005929
3: 0.0158907, 0.0256007, 0.0159921, 0.0256434, -0.0045636, 0.0043733
4: -0.0026401, -0.0019016, -0.0026434, -0.0019093, -0.0003326, 0.0003471
5: 0.0146020, 0.0153484, 0.0145988, 0.0153406, -0.0003362, 0.0003508
6: 0.0044413, 0.0048044, 0.0044451, 0.0048060, -0.0001706, 0.0001635
7: -0.0144129, -0.0118965, -0.0144240, -0.0119227, -0.0011334, 0.0011827
8: 0.0052946, 0.0072910, 0.0052858, 0.0072702, -0.0008992, 0.0009383
9: 0.0072475, 0.0108383, 0.0072317, 0.0108008, -0.0016172, 0.0016876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040764, -0.0041050, -0.0040758, -0.0000142, 0.0000131
1: -0.0064066, -0.0053156, -0.0063873, -0.0052954, -0.0005311, 0.0004919
2: 0.9687753, 0.9700845, 0.9687985, 0.9701087, -0.0006373, 0.0005903
3: 0.0159975, 0.0256543, 0.0161682, 0.0258329, -0.0047008, 0.0043540
4: -0.0026442, -0.0019097, -0.0026578, -0.0019227, -0.0003311, 0.0003575
5: 0.0145979, 0.0153402, 0.0145842, 0.0153271, -0.0003347, 0.0003613
6: 0.0044453, 0.0048064, 0.0044517, 0.0048130, -0.0001758, 0.0001628
7: -0.0144268, -0.0119241, -0.0144731, -0.0119684, -0.0011284, 0.0012182
8: 0.0052836, 0.0072691, 0.0052469, 0.0072340, -0.0008952, 0.0009665
9: 0.0072277, 0.0107988, 0.0071617, 0.0107357, -0.0016101, 0.0017383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004301
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004301
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040765, -0.0041050, -0.0040760, -0.0000145, 0.0000129
1: -0.0064187, -0.0053217, -0.0063864, -0.0053004, -0.0005425, 0.0004845
2: 0.9687607, 0.9700772, 0.9687995, 0.9701028, -0.0006510, 0.0005815
3: 0.0158907, 0.0256007, 0.0161765, 0.0257885, -0.0048016, 0.0042887
4: -0.0026401, -0.0019016, -0.0026544, -0.0019233, -0.0003262, 0.0003652
5: 0.0146020, 0.0153484, 0.0145876, 0.0153265, -0.0003297, 0.0003691
6: 0.0044413, 0.0048044, 0.0044520, 0.0048114, -0.0001795, 0.0001603
7: -0.0144129, -0.0118965, -0.0144616, -0.0119705, -0.0011115, 0.0012444
8: 0.0052946, 0.0072910, 0.0052560, 0.0072323, -0.0008818, 0.0009872
9: 0.0072475, 0.0108383, 0.0071781, 0.0107326, -0.0015860, 0.0017756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004301
time: 0.91 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004301
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041055, -0.0040764, -0.0000131, 0.0000142
1: -0.0063873, -0.0052954, -0.0064066, -0.0053156, -0.0004919, 0.0005311
2: 0.9687985, 0.9701087, 0.9687753, 0.9700845, -0.0005903, 0.0006373
3: 0.0161682, 0.0258329, 0.0159975, 0.0256543, -0.0043540, 0.0047008
4: -0.0026578, -0.0019227, -0.0026442, -0.0019097, -0.0003575, 0.0003311
5: 0.0145842, 0.0153271, 0.0145979, 0.0153402, -0.0003613, 0.0003347
6: 0.0044517, 0.0048130, 0.0044453, 0.0048064, -0.0001628, 0.0001758
7: -0.0144731, -0.0119684, -0.0144268, -0.0119241, -0.0012182, 0.0011284
8: 0.0052469, 0.0072340, 0.0052836, 0.0072691, -0.0009665, 0.0008952
9: 0.0071617, 0.0107357, 0.0072277, 0.0107988, -0.0017383, 0.0016101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040760, -0.0041058, -0.0040765, -0.0000129, 0.0000145
1: -0.0063864, -0.0053004, -0.0064187, -0.0053217, -0.0004845, 0.0005425
2: 0.9687995, 0.9701028, 0.9687607, 0.9700772, -0.0005815, 0.0006510
3: 0.0161765, 0.0257885, 0.0158907, 0.0256007, -0.0042887, 0.0048016
4: -0.0026544, -0.0019233, -0.0026401, -0.0019016, -0.0003652, 0.0003262
5: 0.0145876, 0.0153265, 0.0146020, 0.0153484, -0.0003691, 0.0003297
6: 0.0044520, 0.0048114, 0.0044413, 0.0048044, -0.0001603, 0.0001795
7: -0.0144616, -0.0119705, -0.0144129, -0.0118965, -0.0012444, 0.0011115
8: 0.0052560, 0.0072323, 0.0052946, 0.0072910, -0.0009872, 0.0008818
9: 0.0071781, 0.0107326, 0.0072475, 0.0108383, -0.0017756, 0.0015860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041049, -0.0040759, -0.0041050, -0.0040758, -0.0000135, 0.0000134
1: -0.0063858, -0.0052995, -0.0063873, -0.0052954, -0.0005041, 0.0005013
2: 0.9688003, 0.9701039, 0.9687985, 0.9701087, -0.0006049, 0.0006016
3: 0.0161820, 0.0257973, 0.0161682, 0.0258329, -0.0044615, 0.0044373
4: -0.0026551, -0.0019238, -0.0026578, -0.0019227, -0.0003375, 0.0003393
5: 0.0145869, 0.0153260, 0.0145842, 0.0153271, -0.0003411, 0.0003429
6: 0.0044522, 0.0048117, 0.0044517, 0.0048130, -0.0001668, 0.0001659
7: -0.0144638, -0.0119719, -0.0144731, -0.0119684, -0.0011500, 0.0011562
8: 0.0052542, 0.0072312, 0.0052469, 0.0072340, -0.0009123, 0.0009173
9: 0.0071748, 0.0107306, 0.0071617, 0.0107357, -0.0016409, 0.0016499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004472
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004472
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040761, -0.0041050, -0.0040760, -0.0000138, 0.0000132
1: -0.0063976, -0.0053052, -0.0063864, -0.0053004, -0.0005153, 0.0004937
2: 0.9687861, 0.9700970, 0.9687995, 0.9701028, -0.0006184, 0.0005925
3: 0.0160770, 0.0257465, 0.0161765, 0.0257885, -0.0045614, 0.0043701
4: -0.0026512, -0.0019158, -0.0026544, -0.0019233, -0.0003324, 0.0003469
5: 0.0145908, 0.0153341, 0.0145876, 0.0153265, -0.0003359, 0.0003506
6: 0.0044483, 0.0048098, 0.0044520, 0.0048114, -0.0001705, 0.0001634
7: -0.0144507, -0.0119448, -0.0144616, -0.0119705, -0.0011325, 0.0011821
8: 0.0052647, 0.0072527, 0.0052560, 0.0072323, -0.0008985, 0.0009378
9: 0.0071936, 0.0107694, 0.0071781, 0.0107326, -0.0016160, 0.0016868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004596, upper bound: 0.0004472
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004596, upper bound: 0.0004472
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040766, -0.0041056, -0.0040758, -0.0000145, 0.0000140
1: -0.0063889, -0.0053259, -0.0064088, -0.0052944, -0.0005423, 0.0005226
2: 0.9687966, 0.9700723, 0.9687726, 0.9701099, -0.0006508, 0.0006272
3: 0.0161546, 0.0255636, 0.0159781, 0.0258420, -0.0048001, 0.0046260
4: -0.0026373, -0.0019217, -0.0026585, -0.0019083, -0.0003518, 0.0003651
5: 0.0146049, 0.0153281, 0.0145835, 0.0153417, -0.0003556, 0.0003690
6: 0.0044512, 0.0048030, 0.0044446, 0.0048134, -0.0001795, 0.0001730
7: -0.0144033, -0.0119649, -0.0144754, -0.0119191, -0.0011989, 0.0012440
8: 0.0053023, 0.0072368, 0.0052450, 0.0072731, -0.0009511, 0.0009869
9: 0.0072612, 0.0107407, 0.0071583, 0.0108060, -0.0017107, 0.0017751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040768, -0.0041055, -0.0040759, -0.0000148, 0.0000138
1: -0.0064025, -0.0053311, -0.0064079, -0.0052996, -0.0005541, 0.0005151
2: 0.9687803, 0.9700659, 0.9687737, 0.9701036, -0.0006650, 0.0006182
3: 0.0160343, 0.0255172, 0.0159864, 0.0257956, -0.0049049, 0.0045596
4: -0.0026338, -0.0019125, -0.0026549, -0.0019089, -0.0003468, 0.0003730
5: 0.0146085, 0.0153374, 0.0145871, 0.0153411, -0.0003505, 0.0003770
6: 0.0044467, 0.0048012, 0.0044449, 0.0048116, -0.0001834, 0.0001705
7: -0.0143912, -0.0119337, -0.0144634, -0.0119213, -0.0011817, 0.0012712
8: 0.0053118, 0.0072615, 0.0052546, 0.0072714, -0.0009375, 0.0010085
9: 0.0072784, 0.0107852, 0.0071755, 0.0108029, -0.0016861, 0.0018138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
time: 0.97 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040766, -0.0041050, -0.0040754, -0.0000150, 0.0000135
1: -0.0063889, -0.0053259, -0.0063876, -0.0052785, -0.0005632, 0.0005069
2: 0.9687966, 0.9700723, 0.9687980, 0.9701290, -0.0006759, 0.0006083
3: 0.0161546, 0.0255636, 0.0161655, 0.0259825, -0.0049851, 0.0044871
4: -0.0026373, -0.0019217, -0.0026692, -0.0019225, -0.0003413, 0.0003791
5: 0.0146049, 0.0153281, 0.0145727, 0.0153273, -0.0003449, 0.0003832
6: 0.0044512, 0.0048030, 0.0044516, 0.0048186, -0.0001864, 0.0001678
7: -0.0144033, -0.0119649, -0.0145118, -0.0119677, -0.0011629, 0.0012919
8: 0.0053023, 0.0072368, 0.0052161, 0.0072346, -0.0009226, 0.0010250
9: 0.0072612, 0.0107407, 0.0071063, 0.0107367, -0.0016593, 0.0018435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004371
time: 0.93 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004371
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040768, -0.0041050, -0.0040755, -0.0000154, 0.0000133
1: -0.0064025, -0.0053311, -0.0063867, -0.0052836, -0.0005749, 0.0004994
2: 0.9687803, 0.9700659, 0.9687991, 0.9701228, -0.0006899, 0.0005993
3: 0.0160343, 0.0255172, 0.0161741, 0.0259375, -0.0050886, 0.0044203
4: -0.0026338, -0.0019125, -0.0026657, -0.0019232, -0.0003362, 0.0003870
5: 0.0146085, 0.0153374, 0.0145762, 0.0153266, -0.0003398, 0.0003912
6: 0.0044467, 0.0048012, 0.0044519, 0.0048169, -0.0001903, 0.0001653
7: -0.0143912, -0.0119337, -0.0145002, -0.0119699, -0.0011456, 0.0013188
8: 0.0053118, 0.0072615, 0.0052254, 0.0072328, -0.0009088, 0.0010462
9: 0.0072784, 0.0107852, 0.0071230, 0.0107335, -0.0016346, 0.0018818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004371
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004371
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040762, -0.0041056, -0.0040758, -0.0000142, 0.0000147
1: -0.0063674, -0.0053099, -0.0064088, -0.0052944, -0.0005327, 0.0005496
2: 0.9688223, 0.9700913, 0.9687726, 0.9701099, -0.0006393, 0.0006596
3: 0.0163445, 0.0257047, 0.0159781, 0.0258420, -0.0047154, 0.0048649
4: -0.0026480, -0.0019361, -0.0026585, -0.0019083, -0.0003700, 0.0003586
5: 0.0145940, 0.0153135, 0.0145835, 0.0153417, -0.0003740, 0.0003625
6: 0.0044583, 0.0048082, 0.0044446, 0.0048134, -0.0001763, 0.0001819
7: -0.0144398, -0.0120141, -0.0144754, -0.0119191, -0.0012608, 0.0012220
8: 0.0052733, 0.0071978, 0.0052450, 0.0072731, -0.0010002, 0.0009695
9: 0.0072091, 0.0106705, 0.0071583, 0.0108060, -0.0017990, 0.0017438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040764, -0.0041055, -0.0040759, -0.0000145, 0.0000145
1: -0.0063815, -0.0053153, -0.0064079, -0.0052996, -0.0005440, 0.0005426
2: 0.9688054, 0.9700848, 0.9687737, 0.9701036, -0.0006528, 0.0006512
3: 0.0162200, 0.0256571, 0.0159864, 0.0257956, -0.0048149, 0.0048029
4: -0.0026444, -0.0019267, -0.0026549, -0.0019089, -0.0003653, 0.0003662
5: 0.0145977, 0.0153231, 0.0145871, 0.0153411, -0.0003692, 0.0003701
6: 0.0044536, 0.0048065, 0.0044449, 0.0048116, -0.0001800, 0.0001796
7: -0.0144275, -0.0119818, -0.0144634, -0.0119213, -0.0012447, 0.0012478
8: 0.0052830, 0.0072233, 0.0052546, 0.0072714, -0.0009875, 0.0009900
9: 0.0072267, 0.0107165, 0.0071755, 0.0108029, -0.0017761, 0.0017805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
time: 0.88 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041044, -0.0040762, -0.0041050, -0.0040754, -0.0000145, 0.0000140
1: -0.0063674, -0.0053099, -0.0063876, -0.0052785, -0.0005428, 0.0005227
2: 0.9688223, 0.9700913, 0.9687980, 0.9701290, -0.0006514, 0.0006272
3: 0.0163445, 0.0257047, 0.0161655, 0.0259825, -0.0048044, 0.0046264
4: -0.0026480, -0.0019361, -0.0026692, -0.0019225, -0.0003519, 0.0003654
5: 0.0145940, 0.0153135, 0.0145727, 0.0153273, -0.0003556, 0.0003693
6: 0.0044583, 0.0048082, 0.0044516, 0.0048186, -0.0001796, 0.0001730
7: -0.0144398, -0.0120141, -0.0145118, -0.0119677, -0.0011990, 0.0012451
8: 0.0052733, 0.0071978, 0.0052161, 0.0072346, -0.0009512, 0.0009878
9: 0.0072091, 0.0106705, 0.0071063, 0.0107367, -0.0017108, 0.0017767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004598
time: 0.86 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004606
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040764, -0.0041050, -0.0040755, -0.0000148, 0.0000138
1: -0.0063815, -0.0053153, -0.0063867, -0.0052836, -0.0005548, 0.0005150
2: 0.9688054, 0.9700848, 0.9687991, 0.9701228, -0.0006657, 0.0006181
3: 0.0162200, 0.0256571, 0.0161741, 0.0259375, -0.0049103, 0.0045588
4: -0.0026444, -0.0019267, -0.0026657, -0.0019232, -0.0003467, 0.0003735
5: 0.0145977, 0.0153231, 0.0145762, 0.0153266, -0.0003504, 0.0003774
6: 0.0044536, 0.0048065, 0.0044519, 0.0048169, -0.0001836, 0.0001704
7: -0.0144275, -0.0119818, -0.0145002, -0.0119699, -0.0011815, 0.0012726
8: 0.0052830, 0.0072233, 0.0052254, 0.0072328, -0.0009373, 0.0010096
9: 0.0072267, 0.0107165, 0.0071230, 0.0107335, -0.0016859, 0.0018158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004598
time: 0.90 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004606
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040764, -0.0041056, -0.0040758, -0.0000142, 0.0000136
1: -0.0064066, -0.0053156, -0.0064088, -0.0052944, -0.0005309, 0.0005109
2: 0.9687753, 0.9700845, 0.9687726, 0.9701099, -0.0006371, 0.0006130
3: 0.0159975, 0.0256543, 0.0159781, 0.0258420, -0.0046991, 0.0045217
4: -0.0026442, -0.0019097, -0.0026585, -0.0019083, -0.0003439, 0.0003574
5: 0.0145979, 0.0153402, 0.0145835, 0.0153417, -0.0003476, 0.0003612
6: 0.0044453, 0.0048064, 0.0044446, 0.0048134, -0.0001757, 0.0001691
7: -0.0144268, -0.0119241, -0.0144754, -0.0119191, -0.0011718, 0.0012178
8: 0.0052836, 0.0072691, 0.0052450, 0.0072731, -0.0009297, 0.0009662
9: 0.0072277, 0.0107988, 0.0071583, 0.0108060, -0.0016721, 0.0017377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040765, -0.0041055, -0.0040759, -0.0000145, 0.0000134
1: -0.0064187, -0.0053217, -0.0064079, -0.0052996, -0.0005420, 0.0005034
2: 0.9687607, 0.9700772, 0.9687737, 0.9701036, -0.0006504, 0.0006041
3: 0.0158907, 0.0256007, 0.0159864, 0.0257956, -0.0047976, 0.0044560
4: -0.0026401, -0.0019016, -0.0026549, -0.0019089, -0.0003389, 0.0003649
5: 0.0146020, 0.0153484, 0.0145871, 0.0153411, -0.0003425, 0.0003688
6: 0.0044413, 0.0048044, 0.0044449, 0.0048116, -0.0001794, 0.0001666
7: -0.0144129, -0.0118965, -0.0144634, -0.0119213, -0.0011548, 0.0012433
8: 0.0052946, 0.0072910, 0.0052546, 0.0072714, -0.0009162, 0.0009864
9: 0.0072475, 0.0108383, 0.0071755, 0.0108029, -0.0016478, 0.0017741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040764, -0.0041050, -0.0040754, -0.0000148, 0.0000132
1: -0.0064066, -0.0053156, -0.0063876, -0.0052785, -0.0005524, 0.0004953
2: 0.9687753, 0.9700845, 0.9687980, 0.9701290, -0.0006630, 0.0005944
3: 0.0159975, 0.0256543, 0.0161655, 0.0259825, -0.0048899, 0.0043843
4: -0.0026442, -0.0019097, -0.0026692, -0.0019225, -0.0003335, 0.0003719
5: 0.0145979, 0.0153402, 0.0145727, 0.0153273, -0.0003370, 0.0003759
6: 0.0044453, 0.0048064, 0.0044516, 0.0048186, -0.0001828, 0.0001639
7: -0.0144268, -0.0119241, -0.0145118, -0.0119677, -0.0011362, 0.0012673
8: 0.0052836, 0.0072691, 0.0052161, 0.0072346, -0.0009014, 0.0010054
9: 0.0072277, 0.0107988, 0.0071063, 0.0107367, -0.0016213, 0.0018083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0004254
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0004254
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040765, -0.0041050, -0.0040755, -0.0000151, 0.0000130
1: -0.0064187, -0.0053217, -0.0063867, -0.0052836, -0.0005636, 0.0004879
2: 0.9687607, 0.9700772, 0.9687991, 0.9701228, -0.0006764, 0.0005855
3: 0.0158907, 0.0256007, 0.0161741, 0.0259375, -0.0049890, 0.0043183
4: -0.0026401, -0.0019016, -0.0026657, -0.0019232, -0.0003284, 0.0003794
5: 0.0146020, 0.0153484, 0.0145762, 0.0153266, -0.0003319, 0.0003835
6: 0.0044413, 0.0048044, 0.0044519, 0.0048169, -0.0001865, 0.0001615
7: -0.0144129, -0.0118965, -0.0145002, -0.0119699, -0.0011191, 0.0012930
8: 0.0052946, 0.0072910, 0.0052254, 0.0072328, -0.0008879, 0.0010258
9: 0.0072475, 0.0108383, 0.0071230, 0.0107335, -0.0015969, 0.0018449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0004254
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0004254
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040758, -0.0041055, -0.0040759, -0.0000139, 0.0000144
1: -0.0063873, -0.0052954, -0.0064073, -0.0052983, -0.0005188, 0.0005404
2: 0.9687985, 0.9701087, 0.9687744, 0.9701054, -0.0006225, 0.0006485
3: 0.0161682, 0.0258329, 0.0159915, 0.0258079, -0.0045917, 0.0047835
4: -0.0026578, -0.0019227, -0.0026559, -0.0019093, -0.0003638, 0.0003492
5: 0.0145842, 0.0153271, 0.0145861, 0.0153407, -0.0003677, 0.0003530
6: 0.0044517, 0.0048130, 0.0044451, 0.0048121, -0.0001717, 0.0001788
7: -0.0144731, -0.0119684, -0.0144666, -0.0119226, -0.0012397, 0.0011900
8: 0.0052469, 0.0072340, 0.0052520, 0.0072703, -0.0009835, 0.0009441
9: 0.0071617, 0.0107357, 0.0071709, 0.0108010, -0.0017689, 0.0016980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
time: 0.88 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040760, -0.0041058, -0.0040761, -0.0000137, 0.0000147
1: -0.0063864, -0.0053004, -0.0064180, -0.0053047, -0.0005118, 0.0005498
2: 0.9687995, 0.9701028, 0.9687615, 0.9700977, -0.0006141, 0.0006598
3: 0.0161765, 0.0257885, 0.0158964, 0.0257509, -0.0045297, 0.0048663
4: -0.0026544, -0.0019233, -0.0026515, -0.0019020, -0.0003701, 0.0003445
5: 0.0145876, 0.0153265, 0.0145905, 0.0153480, -0.0003741, 0.0003482
6: 0.0044520, 0.0048114, 0.0044415, 0.0048100, -0.0001694, 0.0001819
7: -0.0144616, -0.0119705, -0.0144518, -0.0118979, -0.0012611, 0.0011739
8: 0.0052560, 0.0072323, 0.0052637, 0.0072899, -0.0010005, 0.0009313
9: 0.0071781, 0.0107326, 0.0071920, 0.0108362, -0.0017995, 0.0016751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041049, -0.0040759, -0.0041050, -0.0040754, -0.0000142, 0.0000136
1: -0.0063858, -0.0052995, -0.0063876, -0.0052785, -0.0005310, 0.0005108
2: 0.9688003, 0.9701039, 0.9687980, 0.9701290, -0.0006372, 0.0006129
3: 0.0161820, 0.0257973, 0.0161655, 0.0259825, -0.0046996, 0.0045210
4: -0.0026551, -0.0019238, -0.0026692, -0.0019225, -0.0003438, 0.0003574
5: 0.0145869, 0.0153260, 0.0145727, 0.0153273, -0.0003475, 0.0003613
6: 0.0044522, 0.0048117, 0.0044516, 0.0048186, -0.0001757, 0.0001690
7: -0.0144638, -0.0119719, -0.0145118, -0.0119677, -0.0011717, 0.0012180
8: 0.0052542, 0.0072312, 0.0052161, 0.0072346, -0.0009295, 0.0009663
9: 0.0071748, 0.0107306, 0.0071063, 0.0107367, -0.0016719, 0.0017379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004586, upper bound: 0.0004458
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004586, upper bound: 0.0004460
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040761, -0.0041050, -0.0040755, -0.0000145, 0.0000134
1: -0.0063976, -0.0053052, -0.0063867, -0.0052836, -0.0005421, 0.0005031
2: 0.9687861, 0.9700970, 0.9687991, 0.9701228, -0.0006505, 0.0006038
3: 0.0160770, 0.0257465, 0.0161741, 0.0259375, -0.0047983, 0.0044534
4: -0.0026512, -0.0019158, -0.0026657, -0.0019232, -0.0003387, 0.0003649
5: 0.0145908, 0.0153341, 0.0145762, 0.0153266, -0.0003423, 0.0003688
6: 0.0044483, 0.0048098, 0.0044519, 0.0048169, -0.0001794, 0.0001665
7: -0.0144507, -0.0119448, -0.0145002, -0.0119699, -0.0011541, 0.0012435
8: 0.0052647, 0.0072527, 0.0052254, 0.0072328, -0.0009156, 0.0009865
9: 0.0071936, 0.0107694, 0.0071230, 0.0107335, -0.0016469, 0.0017744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004458
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004460
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040761, -0.0041050, -0.0040766, -0.0000133, 0.0000138
1: -0.0063893, -0.0053056, -0.0063889, -0.0053259, -0.0004976, 0.0005171
2: 0.9687960, 0.9700965, 0.9687966, 0.9700723, -0.0005972, 0.0006206
3: 0.0161506, 0.0257431, 0.0161546, 0.0255636, -0.0044048, 0.0045774
4: -0.0026509, -0.0019214, -0.0026373, -0.0019217, -0.0003481, 0.0003350
5: 0.0145911, 0.0153285, 0.0146049, 0.0153281, -0.0003519, 0.0003386
6: 0.0044510, 0.0048097, 0.0044512, 0.0048030, -0.0001647, 0.0001711
7: -0.0144498, -0.0119638, -0.0144033, -0.0119649, -0.0011863, 0.0011416
8: 0.0052654, 0.0072376, 0.0053023, 0.0072368, -0.0009411, 0.0009057
9: 0.0071949, 0.0107422, 0.0072612, 0.0107407, -0.0016927, 0.0016289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004162
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004162
time: 0.87 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.16 seconds
IS_A1_B1_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
IS_A1_B1_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
IS_A1_B1_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
IS_A1_B1_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004224, upper bound: 0.0004224
IS_A1_B1_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004277, upper bound: 0.0004305
IS_A1_B1_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004277, upper bound: 0.0004305
IS_A1_B1_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004277, upper bound: 0.0004305
IS_A1_B1_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004277, upper bound: 0.0004305
IS_A1_B1_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
IS_A1_B1_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
IS_A1_B1_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
IS_A1_B1_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004305, upper bound: 0.0004277
IS_A1_B1_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004512, upper bound: 0.0004509
IS_A1_B1_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004512, upper bound: 0.0004509
IS_A1_B1_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004512, upper bound: 0.0004509
IS_A1_B1_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004512, upper bound: 0.0004509
IS_A1_B1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
IS_A1_B1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
IS_A1_B1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
IS_A1_B1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004318, upper bound: 0.0004214
IS_A1_B1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004408, upper bound: 0.0004254
IS_A1_B1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004408, upper bound: 0.0004254
IS_A1_B1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004408, upper bound: 0.0004254
IS_A1_B1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004408, upper bound: 0.0004254
IS_A1_B1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004288
IS_A1_B1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004288
IS_A1_B1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004288
IS_A1_B1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004288
IS_A1_B1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004590, upper bound: 0.0004468
IS_A1_B1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004590, upper bound: 0.0004468
IS_A1_B1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004590, upper bound: 0.0004468
IS_A1_B1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004590, upper bound: 0.0004468
IS_A1_B1_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
IS_A1_B1_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
IS_A1_B1_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
IS_A1_B1_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004162, upper bound: 0.0004144
IS_A1_B1_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004232, upper bound: 0.0004261
IS_A1_B1_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004232, upper bound: 0.0004261
IS_A1_B1_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004232, upper bound: 0.0004261
IS_A1_B1_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004232, upper bound: 0.0004261
IS_A1_B1_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
IS_A1_B1_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
IS_A1_B1_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
IS_A1_B1_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004257, upper bound: 0.0004215
IS_A1_B1_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004504
IS_A1_B1_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004505
IS_A1_B1_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004504
IS_A1_B1_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004505
IS_A1_B1_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
IS_A1_B1_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
IS_A1_B1_B2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
IS_A1_B1_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004128
IS_A1_B1_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004316, upper bound: 0.0004241
IS_A1_B1_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004316, upper bound: 0.0004241
IS_A1_B1_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004316, upper bound: 0.0004241
IS_A1_B1_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004316, upper bound: 0.0004241
IS_A1_B1_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
IS_A1_B1_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
IS_A1_B1_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
IS_A1_B1_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004182
IS_A1_B1_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004585, upper bound: 0.0004455
IS_A1_B1_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004585, upper bound: 0.0004455
IS_A1_B1_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004585, upper bound: 0.0004456
IS_A1_B1_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004585, upper bound: 0.0004456
IS_A1_B2_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
IS_A1_B2_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
IS_A1_B2_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
IS_A1_B2_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004214, upper bound: 0.0004318
IS_A1_B2_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004408
IS_A1_B2_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004408
IS_A1_B2_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004408
IS_A1_B2_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004408
IS_A1_B2_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
IS_A1_B2_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
IS_A1_B2_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
IS_A1_B2_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004288, upper bound: 0.0004367
IS_A1_B2_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004588
IS_A1_B2_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004592
IS_A1_B2_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004588
IS_A1_B2_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004592
IS_A1_B2_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
IS_A1_B2_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
IS_A1_B2_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
IS_A1_B2_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004323, upper bound: 0.0004232
IS_A1_B2_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004301
IS_A1_B2_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004301
IS_A1_B2_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004301
IS_A1_B2_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004301
IS_A1_B2_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
IS_A1_B2_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
IS_A1_B2_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
IS_A1_B2_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004411, upper bound: 0.0004268
IS_A1_B2_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004472
IS_A1_B2_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004472
IS_A1_B2_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004596, upper bound: 0.0004472
IS_A1_B2_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004596, upper bound: 0.0004472
IS_A1_B2_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
IS_A1_B2_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
IS_A1_B2_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
IS_A1_B2_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004251
IS_A1_B2_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004371
IS_A1_B2_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004371
IS_A1_B2_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004371
IS_A1_B2_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004371
IS_A1_B2_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
IS_A1_B2_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
IS_A1_B2_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
IS_A1_B2_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004245, upper bound: 0.0004313
IS_A1_B2_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004598
IS_A1_B2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004606
IS_A1_B2_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004598
IS_A1_B2_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004606
IS_A1_B2_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
IS_A1_B2_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
IS_A1_B2_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
IS_A1_B2_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004143
IS_A1_B2_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0004254
IS_A1_B2_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0004254
IS_A1_B2_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0004254
IS_A1_B2_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0004254
IS_A1_B2_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
IS_A1_B2_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
IS_A1_B2_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
IS_A1_B2_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004193
IS_A1_B2_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004586, upper bound: 0.0004458
IS_A1_B2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004586, upper bound: 0.0004460
IS_A1_B2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004458
IS_A1_B2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004460
IS_A2_B1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004162
IS_A2_B1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.16
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004162
IS_A2_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004162
IS_A2_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004245
IS_A2_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004232
IS_A2_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004181, upper bound: 0.0004352
IS_A2_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004257
IS_A2_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004505
IS_A2_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004144, upper bound: 0.0004505
IS_A2_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004177
IS_A2_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004150
IS_A2_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004219
IS_A2_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004210
IS_A2_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004176
IS_A2_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004245
IS_A2_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004464
IS_A2_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004464
IS_A2_B1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004279
IS_A2_B1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004140, upper bound: 0.0004140
IS_A2_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004351
IS_A2_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004140, upper bound: 0.0004254
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004364, upper bound: 0.0004231
IS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004261, upper bound: 0.0004214
IS_A2_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004264, upper bound: 0.0004514
IS_A2_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004261, upper bound: 0.0004499
IS_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004152
IS_A2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004126
IS_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004190
IS_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004182
IS_A2_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004243
IS_A2_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004239
IS_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004400, upper bound: 0.0004455
IS_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004248, upper bound: 0.0004455
IS_A2_B2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004387
IS_A2_B2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004245
IS_A2_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004258, upper bound: 0.0004329
IS_A2_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004316
IS_A2_B2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004450
IS_A2_B2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004357
IS_A2_B2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004609
IS_A2_B2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004128, upper bound: 0.0004588
IS_A2_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004406, upper bound: 0.0004189
IS_A2_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004162
IS_A2_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004406, upper bound: 0.0004231
IS_A2_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004222
IS_A2_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004342
IS_A2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004257
IS_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004406, upper bound: 0.0004467
IS_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0004467
IS_A2_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004387
IS_A2_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004126, upper bound: 0.0004243
IS_A2_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004450
IS_A2_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004126, upper bound: 0.0004357
IS_A2_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004246, upper bound: 0.0004413
IS_A2_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004241, upper bound: 0.0004311
IS_A2_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004246, upper bound: 0.0004609
IS_A2_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004241, upper bound: 0.0004587
IS_A2_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004283, upper bound: 0.0004275
IS_A2_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004142
IS_A2_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004283, upper bound: 0.0004340
IS_A2_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004251, upper bound: 0.0004251
IS_A2_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004481, upper bound: 0.0004203
IS_A2_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004372, upper bound: 0.0004193
IS_A2_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004377, upper bound: 0.0004477
IS_A2_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 2, lower bound: -0.0004372, upper bound: 0.0004459

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.95 + 598.46 = 601.41 seconds
