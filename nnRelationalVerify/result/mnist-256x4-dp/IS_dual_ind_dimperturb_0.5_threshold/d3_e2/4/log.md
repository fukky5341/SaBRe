## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0051876


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042474, -0.0036923, -0.0042474, -0.0036923, -0.0004486, 0.0004486)
1: (0.0001921, 0.0032656, 0.0001921, 0.0032656, -0.0024840, 0.0024840)
2: (0.0076705, 0.0145370, 0.0076705, 0.0145370, -0.0055495, 0.0055495)
3: (0.0012084, 0.0041020, 0.0012084, 0.0041020, -0.0023386, 0.0023386)
4: (1.0014385, 1.0126644, 1.0014385, 1.0126644, -0.0090728, 0.0090728)
5: (0.0025213, 0.0047051, 0.0025213, 0.0047051, -0.0017650, 0.0017650)
6: (-0.0118660, -0.0090240, -0.0118660, -0.0090240, -0.0022969, 0.0022969)
7: (-0.0103170, -0.0099545, -0.0103170, -0.0099545, -0.0002930, 0.0002930)
8: (-0.0046192, -0.0026556, -0.0046192, -0.0026556, -0.0015870, 0.0015870)
9: (-0.0048761, 0.0049542, -0.0048761, 0.0049542, -0.0079448, 0.0079448)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.80 = 3.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0065175, upper bound: 0.0065175

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 129

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0062699, upper bound: 0.0059108
time: 0.77 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0062699, upper bound: 0.0062699
time: 0.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 4, lower bound: -0.0062699, upper bound: 0.0059108
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 4, lower bound: -0.0062699, upper bound: 0.0062699

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0042096, -0.0037057, -0.0042424, -0.0036925, -0.0004063, 0.0004295
1: 0.0002663, 0.0030564, 0.0001931, 0.0032381, -0.0023783, 0.0022499
2: 0.0081379, 0.0143711, 0.0077320, 0.0145347, -0.0050266, 0.0053134
3: 0.0012783, 0.0039050, 0.0012094, 0.0040761, -0.0022391, 0.0021182
4: 1.0017098, 1.0119004, 1.0014424, 1.0125639, -0.0086868, 0.0082178
5: 0.0025740, 0.0045565, 0.0025220, 0.0046856, -0.0016899, 0.0015987
6: -0.0116726, -0.0090927, -0.0118406, -0.0090250, -0.0020805, 0.0021992
7: -0.0102923, -0.0099632, -0.0103137, -0.0099546, -0.0002654, 0.0002805
8: -0.0045718, -0.0027893, -0.0046186, -0.0026732, -0.0015195, 0.0014374
9: -0.0042070, 0.0047167, -0.0047881, 0.0049508, -0.0071961, 0.0076068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056624, upper bound: 0.0055070
time: 0.81 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058367, upper bound: 0.0054433
time: 0.77 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0042328, -0.0036932, -0.0042474, -0.0036923, -0.0004167, 0.0004479
1: 0.0001972, 0.0031851, 0.0001921, 0.0032656, -0.0024800, 0.0023071
2: 0.0078504, 0.0145257, 0.0076705, 0.0145370, -0.0051544, 0.0055406
3: 0.0012132, 0.0040262, 0.0012084, 0.0041020, -0.0023348, 0.0021721
4: 1.0014571, 1.0123703, 1.0014385, 1.0126644, -0.0090582, 0.0084268
5: 0.0025248, 0.0046479, 0.0025213, 0.0047051, -0.0017622, 0.0016393
6: -0.0117915, -0.0090287, -0.0118660, -0.0090240, -0.0021334, 0.0022932
7: -0.0103075, -0.0099551, -0.0103170, -0.0099545, -0.0002721, 0.0002925
8: -0.0046160, -0.0027071, -0.0046192, -0.0026556, -0.0015844, 0.0014740
9: -0.0046185, 0.0049380, -0.0048761, 0.0049542, -0.0073791, 0.0079320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 129

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059108, upper bound: 0.0062699
time: 0.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059108, upper bound: 0.0062699
time: 1.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 4, lower bound: -0.0056624, upper bound: 0.0055070
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 4, lower bound: -0.0058367, upper bound: 0.0054433
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 4, lower bound: -0.0059108, upper bound: 0.0062699
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 4, lower bound: -0.0059108, upper bound: 0.0062699

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042096, -0.0037057, -0.0042236, -0.0036938, -0.0004054, 0.0004114
1: 0.0002663, 0.0030564, 0.0002004, 0.0031338, -0.0022780, 0.0022445
2: 0.0081379, 0.0143711, 0.0079648, 0.0145184, -0.0050145, 0.0050892
3: 0.0012783, 0.0039050, 0.0012162, 0.0039779, -0.0021446, 0.0021131
4: 1.0017098, 1.0119004, 1.0014689, 1.0121832, -0.0083202, 0.0081981
5: 0.0025740, 0.0045565, 0.0025272, 0.0046115, -0.0016186, 0.0015949
6: -0.0116726, -0.0090927, -0.0117442, -0.0090317, -0.0020755, 0.0021064
7: -0.0102923, -0.0099632, -0.0103014, -0.0099554, -0.0002647, 0.0002687
8: -0.0045718, -0.0027893, -0.0046139, -0.0027398, -0.0014553, 0.0014340
9: -0.0042070, 0.0047167, -0.0044547, 0.0049276, -0.0071789, 0.0072858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056624, upper bound: 0.0052666
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056624, upper bound: 0.0054433
time: 0.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042029, -0.0037062, -0.0042061, -0.0036349, -0.0004589, 0.0004232
1: 0.0002693, 0.0030194, -0.0001256, 0.0030369, -0.0023435, 0.0025407
2: 0.0082205, 0.0143645, 0.0081814, 0.0152467, -0.0056761, 0.0052356
3: 0.0012811, 0.0038702, 0.0009094, 0.0038867, -0.0022063, 0.0023919
4: 1.0017205, 1.0117652, 1.0002784, 1.0118291, -0.0085595, 0.0092798
5: 0.0025761, 0.0045302, 0.0022955, 0.0045426, -0.0016652, 0.0018053
6: -0.0116383, -0.0090954, -0.0116545, -0.0087303, -0.0023493, 0.0021670
7: -0.0102879, -0.0099636, -0.0102900, -0.0099170, -0.0002997, 0.0002764
8: -0.0045699, -0.0028129, -0.0048222, -0.0028018, -0.0014972, 0.0016232
9: -0.0040886, 0.0047072, -0.0041447, 0.0059701, -0.0081261, 0.0074954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049663, upper bound: 0.0039756
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0043009, upper bound: 0.0038201
time: 0.73 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042328, -0.0036932, -0.0042096, -0.0037057, -0.0004200, 0.0004058
1: 0.0001972, 0.0031851, 0.0002663, 0.0030564, -0.0022468, 0.0023257
2: 0.0078504, 0.0145257, 0.0081379, 0.0143711, -0.0051960, 0.0050197
3: 0.0012132, 0.0040262, 0.0012783, 0.0039050, -0.0021153, 0.0021896
4: 1.0014571, 1.0123703, 1.0017098, 1.0119004, -0.0082065, 0.0084948
5: 0.0025248, 0.0046479, 0.0025740, 0.0045565, -0.0015965, 0.0016526
6: -0.0117915, -0.0090287, -0.0116726, -0.0090927, -0.0021506, 0.0020776
7: -0.0103075, -0.0099551, -0.0102923, -0.0099632, -0.0002743, 0.0002650
8: -0.0046160, -0.0027071, -0.0045718, -0.0027893, -0.0014355, 0.0014859
9: -0.0046185, 0.0049380, -0.0042070, 0.0047167, -0.0074387, 0.0071863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055070, upper bound: 0.0056625
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054433, upper bound: 0.0058367
time: 0.79 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042328, -0.0036932, -0.0042328, -0.0036932, -0.0004159, 0.0004159
1: 0.0001972, 0.0031851, 0.0001972, 0.0031851, -0.0023029, 0.0023029
2: 0.0078504, 0.0145257, 0.0078504, 0.0145257, -0.0051449, 0.0051449
3: 0.0012132, 0.0040262, 0.0012132, 0.0040262, -0.0021681, 0.0021681
4: 1.0014571, 1.0123703, 1.0014571, 1.0123703, -0.0084113, 0.0084113
5: 0.0025248, 0.0046479, 0.0025248, 0.0046479, -0.0016363, 0.0016363
6: -0.0117915, -0.0090287, -0.0117915, -0.0090287, -0.0021295, 0.0021295
7: -0.0103075, -0.0099551, -0.0103075, -0.0099551, -0.0002716, 0.0002716
8: -0.0046160, -0.0027071, -0.0046160, -0.0027071, -0.0014713, 0.0014713
9: -0.0046185, 0.0049380, -0.0046185, 0.0049380, -0.0073656, 0.0073656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055070, upper bound: 0.0056625
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054433, upper bound: 0.0058367
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.18 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -0.0056624, upper bound: 0.0052666
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -0.0056624, upper bound: 0.0054433
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 4, lower bound: -0.0049663, upper bound: 0.0039756
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 4, lower bound: -0.0043009, upper bound: 0.0038201
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -0.0055070, upper bound: 0.0056625
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -0.0054433, upper bound: 0.0058367
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -0.0055070, upper bound: 0.0056625
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -0.0054433, upper bound: 0.0058367

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041908, -0.0037067, -0.0042236, -0.0036938, -0.0003874, 0.0003995
1: 0.0002720, 0.0029523, 0.0002004, 0.0031338, -0.0022122, 0.0021451
2: 0.0083704, 0.0143584, 0.0079648, 0.0145184, -0.0047925, 0.0049423
3: 0.0012837, 0.0038070, 0.0012162, 0.0039779, -0.0020827, 0.0020196
4: 1.0017307, 1.0115201, 1.0014689, 1.0121832, -0.0080802, 0.0078351
5: 0.0025781, 0.0044825, 0.0025272, 0.0046115, -0.0015719, 0.0015242
6: -0.0115763, -0.0090979, -0.0117442, -0.0090317, -0.0019836, 0.0020456
7: -0.0102800, -0.0099639, -0.0103014, -0.0099554, -0.0002530, 0.0002609
8: -0.0045682, -0.0028558, -0.0046139, -0.0027398, -0.0014133, 0.0013705
9: -0.0038741, 0.0046984, -0.0044547, 0.0049276, -0.0068610, 0.0070756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044305, upper bound: 0.0047535
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044065, upper bound: 0.0043544
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041729, -0.0036454, -0.0042236, -0.0036938, -0.0003785, 0.0004787
1: -0.0000676, 0.0028530, 0.0002004, 0.0031338, -0.0026507, 0.0020958
2: 0.0085921, 0.0151172, 0.0079648, 0.0145184, -0.0046823, 0.0059219
3: 0.0009639, 0.0037136, 0.0012162, 0.0039779, -0.0024955, 0.0019731
4: 1.0004900, 1.0111576, 1.0014689, 1.0121832, -0.0096815, 0.0076550
5: 0.0023367, 0.0044120, 0.0025272, 0.0046115, -0.0018834, 0.0014892
6: -0.0114845, -0.0087839, -0.0117442, -0.0090317, -0.0019380, 0.0024510
7: -0.0102683, -0.0099238, -0.0103014, -0.0099554, -0.0002472, 0.0003127
8: -0.0047852, -0.0029192, -0.0046139, -0.0027398, -0.0016935, 0.0013390
9: -0.0035566, 0.0057848, -0.0044547, 0.0049276, -0.0067033, 0.0084779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044305, upper bound: 0.0047535
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044065, upper bound: 0.0043544
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0042142, -0.0036945, -0.0042096, -0.0037057, -0.0004027, 0.0003949
1: 0.0002044, 0.0030822, 0.0002663, 0.0030564, -0.0021864, 0.0022298
2: 0.0080802, 0.0145095, 0.0081379, 0.0143711, -0.0049817, 0.0048847
3: 0.0012200, 0.0039293, 0.0012783, 0.0039050, -0.0020584, 0.0020993
4: 1.0014834, 1.0119947, 1.0017098, 1.0119004, -0.0079860, 0.0081445
5: 0.0025300, 0.0045748, 0.0025740, 0.0045565, -0.0015536, 0.0015844
6: -0.0116964, -0.0090354, -0.0116726, -0.0090927, -0.0020619, 0.0020218
7: -0.0102953, -0.0099559, -0.0102923, -0.0099632, -0.0002630, 0.0002579
8: -0.0046114, -0.0027728, -0.0045718, -0.0027893, -0.0013969, 0.0014246
9: -0.0042895, 0.0049148, -0.0042070, 0.0047167, -0.0071319, 0.0069931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052666, upper bound: 0.0056624
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052666, upper bound: 0.0056625
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041952, -0.0036352, -0.0042029, -0.0037062, -0.0004142, 0.0004585
1: -0.0001237, 0.0029770, 0.0002693, 0.0030194, -0.0025387, 0.0022931
2: 0.0083152, 0.0152425, 0.0082205, 0.0143645, -0.0051231, 0.0056717
3: 0.0009111, 0.0038303, 0.0012811, 0.0038702, -0.0023901, 0.0021589
4: 1.0002851, 1.0116103, 1.0017205, 1.0117652, -0.0092726, 0.0083757
5: 0.0022969, 0.0045001, 0.0025761, 0.0045302, -0.0018039, 0.0016294
6: -0.0115991, -0.0087320, -0.0116383, -0.0090954, -0.0021204, 0.0023475
7: -0.0102829, -0.0099172, -0.0102879, -0.0099636, -0.0002705, 0.0002994
8: -0.0048210, -0.0028400, -0.0045699, -0.0028129, -0.0016219, 0.0014650
9: -0.0039530, 0.0059642, -0.0040886, 0.0047072, -0.0073344, 0.0081198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0039755, upper bound: 0.0049663
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0038201, upper bound: 0.0043009
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0042142, -0.0036945, -0.0042328, -0.0036932, -0.0003973, 0.0004063
1: 0.0002044, 0.0030822, 0.0001972, 0.0031851, -0.0022499, 0.0021996
2: 0.0080802, 0.0145095, 0.0078504, 0.0145257, -0.0049142, 0.0050265
3: 0.0012200, 0.0039293, 0.0012132, 0.0040262, -0.0021182, 0.0020708
4: 1.0014834, 1.0119947, 1.0014571, 1.0123703, -0.0082177, 0.0080341
5: 0.0025300, 0.0045748, 0.0025248, 0.0046479, -0.0015987, 0.0015629
6: -0.0116964, -0.0090354, -0.0117915, -0.0090287, -0.0020339, 0.0020804
7: -0.0102953, -0.0099559, -0.0103075, -0.0099551, -0.0002594, 0.0002654
8: -0.0046114, -0.0027728, -0.0046160, -0.0027071, -0.0014374, 0.0014053
9: -0.0042895, 0.0049148, -0.0046185, 0.0049380, -0.0070352, 0.0071961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052868, upper bound: 0.0056625
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052868, upper bound: 0.0056625
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041952, -0.0036352, -0.0042263, -0.0036937, -0.0004088, 0.0004799
1: -0.0001237, 0.0029770, 0.0002001, 0.0031489, -0.0026573, 0.0022634
2: 0.0083152, 0.0152425, 0.0079313, 0.0145190, -0.0050566, 0.0059366
3: 0.0009111, 0.0038303, 0.0012160, 0.0039921, -0.0025017, 0.0021309
4: 1.0002851, 1.0116103, 1.0014679, 1.0122381, -0.0097056, 0.0082669
5: 0.0022969, 0.0045001, 0.0025270, 0.0046222, -0.0018881, 0.0016082
6: -0.0115991, -0.0087320, -0.0117581, -0.0090314, -0.0020929, 0.0024571
7: -0.0102829, -0.0099172, -0.0103032, -0.0099554, -0.0002670, 0.0003134
8: -0.0048210, -0.0028400, -0.0046141, -0.0027302, -0.0016977, 0.0014460
9: -0.0039530, 0.0059642, -0.0045028, 0.0049284, -0.0072392, 0.0084990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041087, upper bound: 0.0049663
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040117, upper bound: 0.0043105
time: 0.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.17 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0044305, upper bound: 0.0047535
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0044065, upper bound: 0.0043544
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0044305, upper bound: 0.0047535
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0044065, upper bound: 0.0043544
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0052666, upper bound: 0.0056624
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0052666, upper bound: 0.0056625
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0039755, upper bound: 0.0049663
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0038201, upper bound: 0.0043009
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0052868, upper bound: 0.0056625
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0052868, upper bound: 0.0056625
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0041087, upper bound: 0.0049663
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -0.0040117, upper bound: 0.0043105

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042142, -0.0036945, -0.0041908, -0.0037067, -0.0003894, 0.0003760
1: 0.0002044, 0.0030822, 0.0002720, 0.0029523, -0.0020819, 0.0021560
2: 0.0080802, 0.0145095, 0.0083704, 0.0143584, -0.0048167, 0.0046513
3: 0.0012200, 0.0039293, 0.0012837, 0.0038070, -0.0019601, 0.0020298
4: 1.0014834, 1.0119947, 1.0017307, 1.0115201, -0.0076043, 0.0078747
5: 0.0025300, 0.0045748, 0.0025781, 0.0044825, -0.0014793, 0.0015319
6: -0.0116964, -0.0090354, -0.0115763, -0.0090979, -0.0019936, 0.0019251
7: -0.0102953, -0.0099559, -0.0102800, -0.0099639, -0.0002543, 0.0002456
8: -0.0046114, -0.0027728, -0.0045682, -0.0028558, -0.0013301, 0.0013774
9: -0.0042895, 0.0049148, -0.0038741, 0.0046984, -0.0068956, 0.0066589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047535, upper bound: 0.0044305
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0043544, upper bound: 0.0044065
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042142, -0.0036945, -0.0041729, -0.0036454, -0.0004686, 0.0003694
1: 0.0002044, 0.0030822, -0.0000676, 0.0028530, -0.0020456, 0.0025944
2: 0.0080802, 0.0145095, 0.0085921, 0.0151172, -0.0057962, 0.0045700
3: 0.0012200, 0.0039293, 0.0009639, 0.0037136, -0.0019258, 0.0024425
4: 1.0014834, 1.0119947, 1.0004900, 1.0111576, -0.0074714, 0.0094760
5: 0.0025300, 0.0045748, 0.0023367, 0.0044120, -0.0014535, 0.0018435
6: -0.0116964, -0.0090354, -0.0114845, -0.0087839, -0.0023990, 0.0018915
7: -0.0102953, -0.0099559, -0.0102683, -0.0099238, -0.0003060, 0.0002413
8: -0.0046114, -0.0027728, -0.0047852, -0.0029192, -0.0013069, 0.0016575
9: -0.0042895, 0.0049148, -0.0035566, 0.0057848, -0.0082979, 0.0065425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047535, upper bound: 0.0044305
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0043544, upper bound: 0.0044065
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042142, -0.0036945, -0.0042142, -0.0036945, -0.0003867, 0.0003867
1: 0.0002044, 0.0030822, 0.0002044, 0.0030822, -0.0021410, 0.0021410
2: 0.0080802, 0.0145095, 0.0080802, 0.0145095, -0.0047832, 0.0047832
3: 0.0012200, 0.0039293, 0.0012200, 0.0039293, -0.0020156, 0.0020156
4: 1.0014834, 1.0119947, 1.0014834, 1.0119947, -0.0078199, 0.0078199
5: 0.0025300, 0.0045748, 0.0025300, 0.0045748, -0.0015213, 0.0015213
6: -0.0116964, -0.0090354, -0.0116964, -0.0090354, -0.0019797, 0.0019797
7: -0.0102953, -0.0099559, -0.0102953, -0.0099559, -0.0002525, 0.0002525
8: -0.0046114, -0.0027728, -0.0046114, -0.0027728, -0.0013678, 0.0013678
9: -0.0042895, 0.0049148, -0.0042895, 0.0049148, -0.0068477, 0.0068477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048330, upper bound: 0.0044386
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044949, upper bound: 0.0044123
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042142, -0.0036945, -0.0041952, -0.0036352, -0.0004658, 0.0003812
1: 0.0002044, 0.0030822, -0.0001237, 0.0029770, -0.0021105, 0.0025791
2: 0.0080802, 0.0145095, 0.0083152, 0.0152425, -0.0057621, 0.0047151
3: 0.0012200, 0.0039293, 0.0009111, 0.0038303, -0.0019869, 0.0024281
4: 1.0014834, 1.0119947, 1.0002851, 1.0116103, -0.0077086, 0.0094203
5: 0.0025300, 0.0045748, 0.0022969, 0.0045001, -0.0014996, 0.0018326
6: -0.0116964, -0.0090354, -0.0115991, -0.0087320, -0.0023849, 0.0019515
7: -0.0102953, -0.0099559, -0.0102829, -0.0099172, -0.0003042, 0.0002489
8: -0.0046114, -0.0027728, -0.0048210, -0.0028400, -0.0013484, 0.0016478
9: -0.0042895, 0.0049148, -0.0039530, 0.0059642, -0.0082491, 0.0067502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048330, upper bound: 0.0044386
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044949, upper bound: 0.0044123
time: 0.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.15 seconds
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 4, lower bound: -0.0047535, upper bound: 0.0044305
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 4, lower bound: -0.0043544, upper bound: 0.0044065
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 4, lower bound: -0.0047535, upper bound: 0.0044305
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 4, lower bound: -0.0043544, upper bound: 0.0044065
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 4, lower bound: -0.0048330, upper bound: 0.0044386
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 4, lower bound: -0.0044949, upper bound: 0.0044123
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 4, lower bound: -0.0048330, upper bound: 0.0044386
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 4, lower bound: -0.0044949, upper bound: 0.0044123

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.39 + 52.88 = 56.27 seconds
