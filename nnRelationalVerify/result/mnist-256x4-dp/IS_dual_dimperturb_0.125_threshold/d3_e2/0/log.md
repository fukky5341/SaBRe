## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00026408


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0011108, -0.0004498, -0.0011108, -0.0004498, -0.0005259, 0.0005259)
1: (-0.0042221, -0.0039942, -0.0042221, -0.0039942, -0.0001922, 0.0001922)
2: (0.0130873, 0.0139861, 0.0130873, 0.0139861, -0.0006893, 0.0006893)
3: (1.0084323, 1.0090191, 1.0084323, 1.0090191, -0.0005869, 0.0005869)
4: (-0.0038715, -0.0037218, -0.0038715, -0.0037218, -0.0001113, 0.0001113)
5: (0.0030938, 0.0036041, 0.0030938, 0.0036041, -0.0004038, 0.0004038)
6: (-0.0024383, -0.0023827, -0.0024383, -0.0023827, -0.0000556, 0.0000556)
7: (-0.0129431, -0.0121192, -0.0129431, -0.0121192, -0.0008091, 0.0008091)
8: (-0.0092759, -0.0076368, -0.0092759, -0.0076368, -0.0012061, 0.0012061)
9: (-0.0005684, 0.0002531, -0.0005684, 0.0002531, -0.0006045, 0.0006045)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.48 = 3.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0003388, upper bound: 0.0003388

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003200, upper bound: 0.0003141
time: 0.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038
time: 0.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 3, lower bound: -0.0003200, upper bound: 0.0003141
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0011107, -0.0004528, -0.0011108, -0.0004498, -0.0005258, 0.0005227
1: -0.0042221, -0.0039956, -0.0042221, -0.0039942, -0.0001921, 0.0001906
2: 0.0130874, 0.0139815, 0.0130873, 0.0139861, -0.0006892, 0.0006843
3: 1.0084356, 1.0090191, 1.0084323, 1.0090191, -0.0005835, 0.0005869
4: -0.0038706, -0.0037218, -0.0038715, -0.0037218, -0.0001104, 0.0001113
5: 0.0030938, 0.0036017, 0.0030938, 0.0036041, -0.0004038, 0.0004013
6: -0.0024381, -0.0023827, -0.0024383, -0.0023827, -0.0000554, 0.0000556
7: -0.0129427, -0.0121192, -0.0129431, -0.0121192, -0.0008087, 0.0008091
8: -0.0092660, -0.0076369, -0.0092759, -0.0076368, -0.0011954, 0.0012059
9: -0.0005683, 0.0002478, -0.0005684, 0.0002531, -0.0006043, 0.0005987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038
time: 0.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038
time: 0.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0011637, -0.0004787, -0.0011108, -0.0004527, -0.0005785, 0.0005217
1: -0.0042337, -0.0040078, -0.0042221, -0.0039955, -0.0002055, 0.0001950
2: 0.0130223, 0.0139417, 0.0130874, 0.0139817, -0.0007562, 0.0006829
3: 1.0084606, 1.0090480, 1.0084352, 1.0090191, -0.0005585, 0.0006127
4: -0.0038632, -0.0037123, -0.0038707, -0.0037218, -0.0001101, 0.0001220
5: 0.0030535, 0.0035814, 0.0030938, 0.0036018, -0.0004440, 0.0004005
6: -0.0024385, -0.0023804, -0.0024382, -0.0023827, -0.0000558, 0.0000578
7: -0.0129397, -0.0120153, -0.0129427, -0.0121192, -0.0008086, 0.0009134
8: -0.0091798, -0.0075415, -0.0092664, -0.0076369, -0.0011921, 0.0013171
9: -0.0006113, 0.0002017, -0.0005683, 0.0002480, -0.0006535, 0.0005969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038
time: 0.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038
time: 0.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.50 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011107, -0.0004528, -0.0011107, -0.0004528, -0.0005226, 0.0005226
1: -0.0042221, -0.0039956, -0.0042221, -0.0039956, -0.0001906, 0.0001906
2: 0.0130874, 0.0139815, 0.0130874, 0.0139815, -0.0006842, 0.0006842
3: 1.0084356, 1.0090191, 1.0084356, 1.0090191, -0.0005835, 0.0005835
4: -0.0038706, -0.0037218, -0.0038706, -0.0037218, -0.0001103, 0.0001103
5: 0.0030938, 0.0036017, 0.0030938, 0.0036017, -0.0004012, 0.0004012
6: -0.0024381, -0.0023827, -0.0024381, -0.0023827, -0.0000554, 0.0000554
7: -0.0129427, -0.0121192, -0.0129427, -0.0121192, -0.0008087, 0.0008087
8: -0.0092660, -0.0076369, -0.0092660, -0.0076369, -0.0011951, 0.0011951
9: -0.0005683, 0.0002478, -0.0005683, 0.0002478, -0.0005985, 0.0005985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003058
time: 0.53 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011107, -0.0004528, -0.0011637, -0.0004787, -0.0005055, 0.0005779
1: -0.0042221, -0.0039956, -0.0042337, -0.0040078, -0.0001847, 0.0002051
2: 0.0130874, 0.0139815, 0.0130223, 0.0139417, -0.0006578, 0.0007552
3: 1.0084356, 1.0090191, 1.0084606, 1.0090480, -0.0006124, 0.0005585
4: -0.0038706, -0.0037218, -0.0038632, -0.0037123, -0.0001218, 0.0001054
5: 0.0030938, 0.0036017, 0.0030535, 0.0035814, -0.0003877, 0.0004435
6: -0.0024381, -0.0023827, -0.0024385, -0.0023804, -0.0000578, 0.0000558
7: -0.0129427, -0.0121192, -0.0129397, -0.0120153, -0.0009133, 0.0008067
8: -0.0092660, -0.0076369, -0.0091798, -0.0075415, -0.0013149, 0.0011379
9: -0.0005683, 0.0002478, -0.0006113, 0.0002017, -0.0005680, 0.0006522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003058
time: 0.50 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011637, -0.0004787, -0.0011107, -0.0004528, -0.0005779, 0.0005055
1: -0.0042337, -0.0040078, -0.0042221, -0.0039956, -0.0002051, 0.0001847
2: 0.0130223, 0.0139417, 0.0130874, 0.0139815, -0.0007552, 0.0006578
3: 1.0084606, 1.0090480, 1.0084356, 1.0090191, -0.0005585, 0.0006124
4: -0.0038632, -0.0037123, -0.0038706, -0.0037218, -0.0001054, 0.0001218
5: 0.0030535, 0.0035814, 0.0030938, 0.0036017, -0.0004435, 0.0003877
6: -0.0024385, -0.0023804, -0.0024381, -0.0023827, -0.0000558, 0.0000578
7: -0.0129397, -0.0120153, -0.0129427, -0.0121192, -0.0008067, 0.0009133
8: -0.0091798, -0.0075415, -0.0092660, -0.0076369, -0.0011379, 0.0013149
9: -0.0006113, 0.0002017, -0.0005683, 0.0002478, -0.0006522, 0.0005680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002971, upper bound: 0.0002925
time: 0.48 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002919, upper bound: 0.0002919
time: 0.50 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011637, -0.0004787, -0.0011637, -0.0004787, -0.0005473, 0.0005473
1: -0.0042337, -0.0040078, -0.0042337, -0.0040078, -0.0001944, 0.0001944
2: 0.0130223, 0.0139417, 0.0130223, 0.0139417, -0.0007059, 0.0007059
3: 1.0084606, 1.0090480, 1.0084606, 1.0090480, -0.0005873, 0.0005873
4: -0.0038632, -0.0037123, -0.0038632, -0.0037123, -0.0001119, 0.0001119
5: 0.0030535, 0.0035814, 0.0030535, 0.0035814, -0.0004193, 0.0004193
6: -0.0024385, -0.0023804, -0.0024385, -0.0023804, -0.0000582, 0.0000582
7: -0.0129397, -0.0120153, -0.0129397, -0.0120153, -0.0009100, 0.0009100
8: -0.0091798, -0.0075415, -0.0091798, -0.0075415, -0.0011997, 0.0011997
9: -0.0006113, 0.0002017, -0.0006113, 0.0002017, -0.0005951, 0.0005951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002932, upper bound: 0.0003025
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038
time: 0.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.68 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003058
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003058
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -0.0002971, upper bound: 0.0002925
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -0.0002919, upper bound: 0.0002919
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -0.0002932, upper bound: 0.0003025
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003038

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004541, -0.0011107, -0.0004528, -0.0005103, 0.0005218
1: -0.0042203, -0.0039964, -0.0042221, -0.0039956, -0.0001881, 0.0001894
2: 0.0131006, 0.0139796, 0.0130874, 0.0139815, -0.0006706, 0.0006829
3: 1.0084382, 1.0090147, 1.0084356, 1.0090191, -0.0005809, 0.0005791
4: -0.0038703, -0.0037236, -0.0038706, -0.0037218, -0.0001101, 0.0001082
5: 0.0031023, 0.0036007, 0.0030938, 0.0036017, -0.0003920, 0.0004005
6: -0.0024374, -0.0023831, -0.0024381, -0.0023827, -0.0000546, 0.0000550
7: -0.0129426, -0.0121485, -0.0129427, -0.0121192, -0.0008086, 0.0007792
8: -0.0092618, -0.0076530, -0.0092660, -0.0076369, -0.0011922, 0.0011754
9: -0.0005617, 0.0002456, -0.0005683, 0.0002478, -0.0005895, 0.0005970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003332, upper bound: 0.0003332
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003332, upper bound: 0.0003341
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010879, -0.0004490, -0.0011076, -0.0004530, -0.0005012, 0.0005309
1: -0.0042186, -0.0039940, -0.0042217, -0.0039958, -0.0001872, 0.0001934
2: 0.0131135, 0.0139874, 0.0130908, 0.0139812, -0.0006610, 0.0006985
3: 1.0084333, 1.0090103, 1.0084362, 1.0090181, -0.0005847, 0.0005741
4: -0.0038717, -0.0037253, -0.0038706, -0.0037223, -0.0001132, 0.0001074
5: 0.0031110, 0.0036047, 0.0030962, 0.0036016, -0.0003852, 0.0004079
6: -0.0024372, -0.0023835, -0.0024380, -0.0023828, -0.0000544, 0.0000545
7: -0.0129432, -0.0121804, -0.0129427, -0.0121272, -0.0008021, 0.0007479
8: -0.0092787, -0.0076694, -0.0092652, -0.0076412, -0.0012293, 0.0011684
9: -0.0005552, 0.0002546, -0.0005666, 0.0002474, -0.0005869, 0.0006175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003341, upper bound: 0.0003332
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003341, upper bound: 0.0003341
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004541, -0.0011637, -0.0004787, -0.0004931, 0.0005770
1: -0.0042203, -0.0039964, -0.0042337, -0.0040078, -0.0001823, 0.0002039
2: 0.0131006, 0.0139796, 0.0130223, 0.0139417, -0.0006442, 0.0007538
3: 1.0084382, 1.0090147, 1.0084606, 1.0090480, -0.0006098, 0.0005541
4: -0.0038703, -0.0037236, -0.0038632, -0.0037123, -0.0001215, 0.0001033
5: 0.0031023, 0.0036007, 0.0030535, 0.0035814, -0.0003784, 0.0004428
6: -0.0024374, -0.0023831, -0.0024385, -0.0023804, -0.0000570, 0.0000554
7: -0.0129426, -0.0121485, -0.0129397, -0.0120153, -0.0009132, 0.0007772
8: -0.0092618, -0.0076530, -0.0091798, -0.0075415, -0.0013120, 0.0011182
9: -0.0005617, 0.0002456, -0.0006113, 0.0002017, -0.0005589, 0.0006507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003104, upper bound: 0.0002966
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010879, -0.0004490, -0.0011606, -0.0004789, -0.0004840, 0.0005861
1: -0.0042186, -0.0039940, -0.0042333, -0.0040080, -0.0001814, 0.0002078
2: 0.0131135, 0.0139874, 0.0130257, 0.0139414, -0.0006346, 0.0007693
3: 1.0084333, 1.0090103, 1.0084614, 1.0090469, -0.0006136, 0.0005490
4: -0.0038717, -0.0037253, -0.0038632, -0.0037127, -0.0001246, 0.0001025
5: 0.0031110, 0.0036047, 0.0030559, 0.0035812, -0.0003717, 0.0004501
6: -0.0024372, -0.0023835, -0.0024383, -0.0023805, -0.0000568, 0.0000549
7: -0.0129432, -0.0121804, -0.0129396, -0.0120232, -0.0009068, 0.0007458
8: -0.0092787, -0.0076694, -0.0091790, -0.0075458, -0.0013486, 0.0011112
9: -0.0005552, 0.0002546, -0.0006095, 0.0002013, -0.0005563, 0.0006708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003044, upper bound: 0.0003047
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003058
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011637, -0.0004787, -0.0010995, -0.0004541, -0.0005770, 0.0004931
1: -0.0042337, -0.0040078, -0.0042203, -0.0039964, -0.0002039, 0.0001823
2: 0.0130223, 0.0139417, 0.0131006, 0.0139796, -0.0007538, 0.0006442
3: 1.0084606, 1.0090480, 1.0084382, 1.0090147, -0.0005541, 0.0006098
4: -0.0038632, -0.0037123, -0.0038703, -0.0037236, -0.0001033, 0.0001215
5: 0.0030535, 0.0035814, 0.0031023, 0.0036007, -0.0004428, 0.0003784
6: -0.0024385, -0.0023804, -0.0024374, -0.0023831, -0.0000554, 0.0000570
7: -0.0129397, -0.0120153, -0.0129426, -0.0121485, -0.0007772, 0.0009132
8: -0.0091798, -0.0075415, -0.0092618, -0.0076530, -0.0011182, 0.0013120
9: -0.0006113, 0.0002017, -0.0005617, 0.0002456, -0.0006507, 0.0005589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003104
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003097, upper bound: 0.0003122
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011606, -0.0004789, -0.0010879, -0.0004490, -0.0005861, 0.0004840
1: -0.0042333, -0.0040080, -0.0042186, -0.0039940, -0.0002078, 0.0001814
2: 0.0130257, 0.0139414, 0.0131135, 0.0139874, -0.0007693, 0.0006346
3: 1.0084614, 1.0090469, 1.0084333, 1.0090103, -0.0005490, 0.0006136
4: -0.0038632, -0.0037127, -0.0038717, -0.0037253, -0.0001025, 0.0001246
5: 0.0030559, 0.0035812, 0.0031110, 0.0036047, -0.0004501, 0.0003717
6: -0.0024383, -0.0023805, -0.0024372, -0.0023835, -0.0000549, 0.0000568
7: -0.0129396, -0.0120232, -0.0129432, -0.0121804, -0.0007458, 0.0009068
8: -0.0091790, -0.0075458, -0.0092787, -0.0076694, -0.0011112, 0.0013486
9: -0.0006095, 0.0002013, -0.0005552, 0.0002546, -0.0006708, 0.0005563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003047, upper bound: 0.0003044
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003058, upper bound: 0.0003122
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011083, -0.0004693, -0.0011566, -0.0004793, -0.0004832, 0.0005299
1: -0.0042263, -0.0040047, -0.0042329, -0.0040082, -0.0001838, 0.0001754
2: 0.0130850, 0.0139562, 0.0130300, 0.0139408, -0.0006335, 0.0006813
3: 1.0084635, 1.0090295, 1.0084624, 1.0090460, -0.0005477, 0.0005671
4: -0.0038659, -0.0037205, -0.0038631, -0.0037132, -0.0001073, 0.0001023
5: 0.0030952, 0.0035888, 0.0030588, 0.0035809, -0.0003709, 0.0004058
6: -0.0024340, -0.0023819, -0.0024378, -0.0023805, -0.0000534, 0.0000560
7: -0.0129408, -0.0121795, -0.0129396, -0.0120354, -0.0008887, 0.0007453
8: -0.0092111, -0.0076158, -0.0091778, -0.0075501, -0.0011385, 0.0011111
9: -0.0005833, 0.0002185, -0.0006081, 0.0002007, -0.0005582, 0.0005546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002799, upper bound: 0.0002948
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002765, upper bound: 0.0002884
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011213, -0.0004799, -0.0011637, -0.0004787, -0.0004944, 0.0005466
1: -0.0042272, -0.0040087, -0.0042337, -0.0040078, -0.0001828, 0.0001900
2: 0.0130706, 0.0139399, 0.0130223, 0.0139417, -0.0006400, 0.0007049
3: 1.0084666, 1.0090318, 1.0084606, 1.0090480, -0.0005814, 0.0005711
4: -0.0038629, -0.0037187, -0.0038632, -0.0037123, -0.0001117, 0.0001017
5: 0.0030854, 0.0035804, 0.0030535, 0.0035814, -0.0003789, 0.0004188
6: -0.0024360, -0.0023817, -0.0024385, -0.0023804, -0.0000556, 0.0000568
7: -0.0129395, -0.0121160, -0.0129397, -0.0120153, -0.0009099, 0.0008097
8: -0.0091757, -0.0076015, -0.0091798, -0.0075415, -0.0011974, 0.0010975
9: -0.0005868, 0.0001995, -0.0006113, 0.0002017, -0.0005478, 0.0005939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002925, upper bound: 0.0002971
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002919, upper bound: 0.0002919
time: 0.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.77 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003332, upper bound: 0.0003332
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003332, upper bound: 0.0003341
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003341, upper bound: 0.0003332
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003341, upper bound: 0.0003341
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003104, upper bound: 0.0002966
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003044, upper bound: 0.0003047
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003058
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003104
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003097, upper bound: 0.0003122
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003047, upper bound: 0.0003044
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0003058, upper bound: 0.0003122
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0002799, upper bound: 0.0002948
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0002765, upper bound: 0.0002884
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0002925, upper bound: 0.0002971
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -0.0002919, upper bound: 0.0002919

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004541, -0.0010995, -0.0004541, -0.0005094, 0.0005094
1: -0.0042203, -0.0039964, -0.0042203, -0.0039964, -0.0001869, 0.0001869
2: 0.0131006, 0.0139796, 0.0131006, 0.0139796, -0.0006692, 0.0006692
3: 1.0084382, 1.0090147, 1.0084382, 1.0090147, -0.0005765, 0.0005765
4: -0.0038703, -0.0037236, -0.0038703, -0.0037236, -0.0001080, 0.0001080
5: 0.0031023, 0.0036007, 0.0031023, 0.0036007, -0.0003913, 0.0003913
6: -0.0024374, -0.0023831, -0.0024374, -0.0023831, -0.0000543, 0.0000543
7: -0.0129426, -0.0121485, -0.0129426, -0.0121485, -0.0007791, 0.0007791
8: -0.0092618, -0.0076530, -0.0092618, -0.0076530, -0.0011725, 0.0011725
9: -0.0005617, 0.0002456, -0.0005617, 0.0002456, -0.0005879, 0.0005879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002665, upper bound: 0.0002969
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003324, upper bound: 0.0003344
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004541, -0.0010879, -0.0004490, -0.0005215, 0.0005005
1: -0.0042203, -0.0039964, -0.0042186, -0.0039940, -0.0001912, 0.0001872
2: 0.0131006, 0.0139796, 0.0131135, 0.0139874, -0.0006879, 0.0006609
3: 1.0084382, 1.0090147, 1.0084333, 1.0090103, -0.0005721, 0.0005814
4: -0.0038703, -0.0037236, -0.0038717, -0.0037253, -0.0001075, 0.0001114
5: 0.0031023, 0.0036007, 0.0031110, 0.0036047, -0.0004008, 0.0003847
6: -0.0024374, -0.0023831, -0.0024372, -0.0023835, -0.0000539, 0.0000541
7: -0.0129426, -0.0121485, -0.0129432, -0.0121804, -0.0007475, 0.0007806
8: -0.0092618, -0.0076530, -0.0092787, -0.0076694, -0.0011691, 0.0012129
9: -0.0005617, 0.0002456, -0.0005552, 0.0002546, -0.0006095, 0.0005888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003319, upper bound: 0.0003341
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003321, upper bound: 0.0003342
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010879, -0.0004490, -0.0010995, -0.0004541, -0.0005005, 0.0005215
1: -0.0042186, -0.0039940, -0.0042203, -0.0039964, -0.0001872, 0.0001912
2: 0.0131135, 0.0139874, 0.0131006, 0.0139796, -0.0006609, 0.0006879
3: 1.0084333, 1.0090103, 1.0084382, 1.0090147, -0.0005814, 0.0005721
4: -0.0038717, -0.0037253, -0.0038703, -0.0037236, -0.0001114, 0.0001075
5: 0.0031110, 0.0036047, 0.0031023, 0.0036007, -0.0003847, 0.0004008
6: -0.0024372, -0.0023835, -0.0024374, -0.0023831, -0.0000541, 0.0000539
7: -0.0129432, -0.0121804, -0.0129426, -0.0121485, -0.0007806, 0.0007475
8: -0.0092787, -0.0076694, -0.0092618, -0.0076530, -0.0012129, 0.0011691
9: -0.0005552, 0.0002546, -0.0005617, 0.0002456, -0.0005888, 0.0006095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003320
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003321
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010879, -0.0004490, -0.0010879, -0.0004490, -0.0005059, 0.0005059
1: -0.0042186, -0.0039940, -0.0042186, -0.0039940, -0.0001875, 0.0001875
2: 0.0131135, 0.0139874, 0.0131135, 0.0139874, -0.0006683, 0.0006683
3: 1.0084333, 1.0090103, 1.0084333, 1.0090103, -0.0005755, 0.0005755
4: -0.0038717, -0.0037253, -0.0038717, -0.0037253, -0.0001087, 0.0001087
5: 0.0031110, 0.0036047, 0.0031110, 0.0036047, -0.0003889, 0.0003889
6: -0.0024372, -0.0023835, -0.0024372, -0.0023835, -0.0000538, 0.0000538
7: -0.0129432, -0.0121804, -0.0129432, -0.0121804, -0.0007484, 0.0007484
8: -0.0092787, -0.0076694, -0.0092787, -0.0076694, -0.0011842, 0.0011842
9: -0.0005552, 0.0002546, -0.0005552, 0.0002546, -0.0005953, 0.0005953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003320
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003321
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010929, -0.0004546, -0.0011083, -0.0004693, -0.0004794, 0.0005144
1: -0.0042195, -0.0039967, -0.0042263, -0.0040047, -0.0001640, 0.0001937
2: 0.0131077, 0.0139787, 0.0130850, 0.0139562, -0.0006248, 0.0006842
3: 1.0084395, 1.0090127, 1.0084635, 1.0090295, -0.0005900, 0.0005087
4: -0.0038701, -0.0037245, -0.0038659, -0.0037205, -0.0001119, 0.0000998
5: 0.0031072, 0.0036003, 0.0030952, 0.0035888, -0.0003677, 0.0003958
6: -0.0024368, -0.0023833, -0.0024340, -0.0023819, -0.0000549, 0.0000507
7: -0.0129425, -0.0121686, -0.0129408, -0.0121795, -0.0007486, 0.0007568
8: -0.0092598, -0.0076614, -0.0092111, -0.0076158, -0.0012208, 0.0010681
9: -0.0005586, 0.0002445, -0.0005833, 0.0002185, -0.0005246, 0.0006147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003104, upper bound: 0.0002966
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003104, upper bound: 0.0002966
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004541, -0.0011213, -0.0004799, -0.0004924, 0.0005241
1: -0.0042203, -0.0039964, -0.0042272, -0.0040087, -0.0001781, 0.0001925
2: 0.0131006, 0.0139796, 0.0130706, 0.0139399, -0.0006430, 0.0006880
3: 1.0084382, 1.0090147, 1.0084666, 1.0090318, -0.0005935, 0.0005481
4: -0.0038703, -0.0037236, -0.0038629, -0.0037187, -0.0001117, 0.0001031
5: 0.0031023, 0.0036007, 0.0030854, 0.0035804, -0.0003779, 0.0004024
6: -0.0024374, -0.0023831, -0.0024360, -0.0023817, -0.0000557, 0.0000529
7: -0.0129426, -0.0121485, -0.0129395, -0.0121160, -0.0008130, 0.0007771
8: -0.0092618, -0.0076530, -0.0091757, -0.0076015, -0.0012103, 0.0011158
9: -0.0005617, 0.0002456, -0.0005868, 0.0001995, -0.0005576, 0.0006042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010299, -0.0004405, -0.0011535, -0.0004795, -0.0004166, 0.0005671
1: -0.0042113, -0.0039914, -0.0042324, -0.0040083, -0.0001710, 0.0001897
2: 0.0131774, 0.0140004, 0.0130335, 0.0139405, -0.0005591, 0.0007419
3: 1.0084351, 1.0089922, 1.0084630, 1.0090449, -0.0005496, 0.0005292
4: -0.0038741, -0.0037335, -0.0038630, -0.0037137, -0.0001195, 0.0000929
5: 0.0031545, 0.0036114, 0.0030612, 0.0035807, -0.0003209, 0.0004353
6: -0.0024334, -0.0023850, -0.0024376, -0.0023806, -0.0000528, 0.0000527
7: -0.0129441, -0.0123540, -0.0129396, -0.0120433, -0.0008853, 0.0005717
8: -0.0093068, -0.0077432, -0.0091771, -0.0075544, -0.0012866, 0.0010227
9: -0.0005275, 0.0002696, -0.0006064, 0.0002003, -0.0005197, 0.0006336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003026
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003036, upper bound: 0.0003046
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010465, -0.0004503, -0.0011606, -0.0004789, -0.0004326, 0.0005854
1: -0.0042120, -0.0039950, -0.0042333, -0.0040080, -0.0001708, 0.0002040
2: 0.0131608, 0.0139853, 0.0130257, 0.0139414, -0.0005706, 0.0007681
3: 1.0084380, 1.0089939, 1.0084614, 1.0090469, -0.0005949, 0.0005325
4: -0.0038713, -0.0037317, -0.0038632, -0.0037127, -0.0001244, 0.0000928
5: 0.0031422, 0.0036037, 0.0030559, 0.0035812, -0.0003325, 0.0004495
6: -0.0024350, -0.0023848, -0.0024383, -0.0023805, -0.0000546, 0.0000535
7: -0.0129430, -0.0122802, -0.0129396, -0.0120232, -0.0009067, 0.0006464
8: -0.0092742, -0.0077300, -0.0091790, -0.0075458, -0.0013461, 0.0010154
9: -0.0005303, 0.0002522, -0.0006095, 0.0002013, -0.0005122, 0.0006694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003044
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003056
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011083, -0.0004693, -0.0010929, -0.0004546, -0.0005144, 0.0004794
1: -0.0042263, -0.0040047, -0.0042195, -0.0039967, -0.0001937, 0.0001640
2: 0.0130850, 0.0139562, 0.0131077, 0.0139787, -0.0006842, 0.0006248
3: 1.0084635, 1.0090295, 1.0084395, 1.0090127, -0.0005087, 0.0005900
4: -0.0038659, -0.0037205, -0.0038701, -0.0037245, -0.0000998, 0.0001119
5: 0.0030952, 0.0035888, 0.0031072, 0.0036003, -0.0003958, 0.0003677
6: -0.0024340, -0.0023819, -0.0024368, -0.0023833, -0.0000507, 0.0000549
7: -0.0129408, -0.0121795, -0.0129425, -0.0121686, -0.0007568, 0.0007486
8: -0.0092111, -0.0076158, -0.0092598, -0.0076614, -0.0010681, 0.0012208
9: -0.0005833, 0.0002185, -0.0005586, 0.0002445, -0.0006147, 0.0005246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003104
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003104
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011213, -0.0004799, -0.0010995, -0.0004541, -0.0005241, 0.0004924
1: -0.0042272, -0.0040087, -0.0042203, -0.0039964, -0.0001925, 0.0001781
2: 0.0130706, 0.0139399, 0.0131006, 0.0139796, -0.0006880, 0.0006430
3: 1.0084666, 1.0090318, 1.0084382, 1.0090147, -0.0005481, 0.0005935
4: -0.0038629, -0.0037187, -0.0038703, -0.0037236, -0.0001031, 0.0001117
5: 0.0030854, 0.0035804, 0.0031023, 0.0036007, -0.0004024, 0.0003779
6: -0.0024360, -0.0023817, -0.0024374, -0.0023831, -0.0000529, 0.0000557
7: -0.0129395, -0.0121160, -0.0129426, -0.0121485, -0.0007771, 0.0008130
8: -0.0091757, -0.0076015, -0.0092618, -0.0076530, -0.0011158, 0.0012103
9: -0.0005868, 0.0001995, -0.0005617, 0.0002456, -0.0006042, 0.0005576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003097, upper bound: 0.0003122
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003097, upper bound: 0.0003122
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011535, -0.0004795, -0.0010299, -0.0004405, -0.0005671, 0.0004166
1: -0.0042324, -0.0040083, -0.0042113, -0.0039914, -0.0001897, 0.0001710
2: 0.0130335, 0.0139405, 0.0131774, 0.0140004, -0.0007419, 0.0005591
3: 1.0084630, 1.0090449, 1.0084351, 1.0089922, -0.0005292, 0.0005496
4: -0.0038630, -0.0037137, -0.0038741, -0.0037335, -0.0000929, 0.0001195
5: 0.0030612, 0.0035807, 0.0031545, 0.0036114, -0.0004353, 0.0003209
6: -0.0024376, -0.0023806, -0.0024334, -0.0023850, -0.0000527, 0.0000528
7: -0.0129396, -0.0120433, -0.0129441, -0.0123540, -0.0005717, 0.0008853
8: -0.0091771, -0.0075544, -0.0093068, -0.0077432, -0.0010227, 0.0012866
9: -0.0006064, 0.0002003, -0.0005275, 0.0002696, -0.0006336, 0.0005197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003046, upper bound: 0.0003036
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011606, -0.0004789, -0.0010465, -0.0004503, -0.0005854, 0.0004326
1: -0.0042333, -0.0040080, -0.0042120, -0.0039950, -0.0002040, 0.0001708
2: 0.0130257, 0.0139414, 0.0131608, 0.0139853, -0.0007681, 0.0005706
3: 1.0084614, 1.0090469, 1.0084380, 1.0089939, -0.0005325, 0.0005949
4: -0.0038632, -0.0037127, -0.0038713, -0.0037317, -0.0000928, 0.0001244
5: 0.0030559, 0.0035812, 0.0031422, 0.0036037, -0.0004495, 0.0003325
6: -0.0024383, -0.0023805, -0.0024350, -0.0023848, -0.0000535, 0.0000546
7: -0.0129396, -0.0120232, -0.0129430, -0.0122802, -0.0006464, 0.0009067
8: -0.0091790, -0.0075458, -0.0092742, -0.0077300, -0.0010154, 0.0013461
9: -0.0006095, 0.0002013, -0.0005303, 0.0002522, -0.0006694, 0.0005122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003044, upper bound: 0.0003118
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003056, upper bound: 0.0003118
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010975, -0.0004704, -0.0011566, -0.0004793, -0.0004728, 0.0005291
1: -0.0042245, -0.0040053, -0.0042329, -0.0040082, -0.0001816, 0.0001743
2: 0.0130971, 0.0139545, 0.0130300, 0.0139408, -0.0006215, 0.0006800
3: 1.0084658, 1.0090250, 1.0084624, 1.0090460, -0.0005405, 0.0005625
4: -0.0038656, -0.0037222, -0.0038631, -0.0037132, -0.0001070, 0.0001004
5: 0.0031033, 0.0035879, 0.0030588, 0.0035809, -0.0003631, 0.0004051
6: -0.0024333, -0.0023822, -0.0024378, -0.0023805, -0.0000527, 0.0000556
7: -0.0129406, -0.0122074, -0.0129396, -0.0120354, -0.0008886, 0.0007174
8: -0.0092075, -0.0076320, -0.0091778, -0.0075501, -0.0011357, 0.0010929
9: -0.0005766, 0.0002165, -0.0006081, 0.0002007, -0.0005500, 0.0005531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002772, upper bound: 0.0002921
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002792, upper bound: 0.0002945
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010833, -0.0004651, -0.0011535, -0.0004795, -0.0004608, 0.0005385
1: -0.0042228, -0.0040032, -0.0042324, -0.0040083, -0.0001808, 0.0001776
2: 0.0131129, 0.0139626, 0.0130335, 0.0139405, -0.0006097, 0.0006959
3: 1.0084625, 1.0090209, 1.0084630, 1.0090449, -0.0005408, 0.0005579
4: -0.0038671, -0.0037241, -0.0038630, -0.0037137, -0.0001102, 0.0000994
5: 0.0031139, 0.0035921, 0.0030612, 0.0035807, -0.0003542, 0.0004127
6: -0.0024330, -0.0023826, -0.0024376, -0.0023806, -0.0000524, 0.0000550
7: -0.0129413, -0.0122432, -0.0129396, -0.0120433, -0.0008821, 0.0006814
8: -0.0092250, -0.0076480, -0.0091771, -0.0075544, -0.0011736, 0.0010850
9: -0.0005702, 0.0002259, -0.0006064, 0.0002003, -0.0005476, 0.0005742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002722, upper bound: 0.0002844
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002878
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0011097, -0.0004812, -0.0011637, -0.0004787, -0.0004817, 0.0005458
1: -0.0042254, -0.0040094, -0.0042337, -0.0040078, -0.0001805, 0.0001888
2: 0.0130838, 0.0139379, 0.0130223, 0.0139417, -0.0006265, 0.0007036
3: 1.0084695, 1.0090274, 1.0084606, 1.0090480, -0.0005785, 0.0005667
4: -0.0038625, -0.0037205, -0.0038632, -0.0037123, -0.0001115, 0.0000998
5: 0.0030942, 0.0035794, 0.0030535, 0.0035814, -0.0003694, 0.0004181
6: -0.0024351, -0.0023821, -0.0024385, -0.0023804, -0.0000548, 0.0000565
7: -0.0129394, -0.0121469, -0.0129397, -0.0120153, -0.0009098, 0.0007783
8: -0.0091715, -0.0076180, -0.0091798, -0.0075415, -0.0011946, 0.0010785
9: -0.0005802, 0.0001973, -0.0006113, 0.0002017, -0.0005395, 0.0005924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002812
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002911
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010987, -0.0004762, -0.0011606, -0.0004789, -0.0004738, 0.0005548
1: -0.0042236, -0.0040070, -0.0042333, -0.0040080, -0.0001796, 0.0001930
2: 0.0130964, 0.0139456, 0.0130257, 0.0139414, -0.0006173, 0.0007189
3: 1.0084643, 1.0090227, 1.0084614, 1.0090469, -0.0005826, 0.0005614
4: -0.0038639, -0.0037222, -0.0038632, -0.0037127, -0.0001145, 0.0000988
5: 0.0031025, 0.0035833, 0.0030559, 0.0035812, -0.0003635, 0.0004253
6: -0.0024351, -0.0023824, -0.0024383, -0.0023805, -0.0000546, 0.0000559
7: -0.0129400, -0.0121725, -0.0129396, -0.0120232, -0.0009034, 0.0007529
8: -0.0091881, -0.0076340, -0.0091790, -0.0075458, -0.0012310, 0.0010719
9: -0.0005734, 0.0002062, -0.0006095, 0.0002013, -0.0005369, 0.0006125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002917, upper bound: 0.0002902
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002914, upper bound: 0.0002914
time: 0.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002665, upper bound: 0.0002969
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003324, upper bound: 0.0003344
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003319, upper bound: 0.0003341
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003321, upper bound: 0.0003342
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003320
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003321
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003320
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003321
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003104, upper bound: 0.0002966
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003104, upper bound: 0.0002966
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003122, upper bound: 0.0003097
IS_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003026
IS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003036, upper bound: 0.0003046
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003044
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003056
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003104
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003104
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003097, upper bound: 0.0003122
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003097, upper bound: 0.0003122
IS_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
IS_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003046, upper bound: 0.0003036
IS_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003044, upper bound: 0.0003118
IS_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0003056, upper bound: 0.0003118
IS_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002772, upper bound: 0.0002921
IS_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002792, upper bound: 0.0002945
IS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002722, upper bound: 0.0002844
IS_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002878
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002812
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002911
IS_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002917, upper bound: 0.0002902
IS_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 3, lower bound: -0.0002914, upper bound: 0.0002914

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011323, -0.0005051, -0.0010995, -0.0004566, -0.0005395, 0.0004656
1: -0.0042285, -0.0040199, -0.0042203, -0.0039976, -0.0001962, 0.0001637
2: 0.0130606, 0.0139011, 0.0131006, 0.0139757, -0.0007058, 0.0006003
3: 1.0084885, 1.0090350, 1.0084409, 1.0090147, -0.0005262, 0.0005941
4: -0.0038556, -0.0037176, -0.0038695, -0.0037236, -0.0000948, 0.0001141
5: 0.0030775, 0.0035605, 0.0031023, 0.0035987, -0.0004139, 0.0003567
6: -0.0024356, -0.0023814, -0.0024372, -0.0023831, -0.0000525, 0.0000558
7: -0.0129366, -0.0120880, -0.0129423, -0.0121485, -0.0007743, 0.0008391
8: -0.0090917, -0.0075900, -0.0092533, -0.0076530, -0.0010062, 0.0012375
9: -0.0005918, 0.0001546, -0.0005617, 0.0002410, -0.0006213, 0.0004899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002487, upper bound: 0.0002487
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002487, upper bound: 0.0002987
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004549, -0.0010995, -0.0004541, -0.0005094, 0.0005078
1: -0.0042203, -0.0039968, -0.0042203, -0.0039964, -0.0001868, 0.0001907
2: 0.0131006, 0.0139783, 0.0131006, 0.0139796, -0.0006692, 0.0006673
3: 1.0084393, 1.0090147, 1.0084382, 1.0090147, -0.0005754, 0.0005765
4: -0.0038700, -0.0037236, -0.0038703, -0.0037236, -0.0001088, 0.0001079
5: 0.0031023, 0.0036001, 0.0031023, 0.0036007, -0.0003913, 0.0003898
6: -0.0024373, -0.0023831, -0.0024374, -0.0023831, -0.0000542, 0.0000543
7: -0.0129425, -0.0121485, -0.0129426, -0.0121485, -0.0007790, 0.0007791
8: -0.0092589, -0.0076530, -0.0092618, -0.0076530, -0.0011866, 0.0011720
9: -0.0005617, 0.0002440, -0.0005617, 0.0002456, -0.0005875, 0.0005988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0002712
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0003351
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0010983, -0.0004542, -0.0010706, -0.0004518, -0.0005153, 0.0004820
1: -0.0042202, -0.0039965, -0.0042161, -0.0039955, -0.0001865, 0.0001835
2: 0.0131019, 0.0139793, 0.0131326, 0.0139830, -0.0006789, 0.0006410
3: 1.0084385, 1.0090144, 1.0084379, 1.0090041, -0.0005656, 0.0005629
4: -0.0038702, -0.0037237, -0.0038709, -0.0037278, -0.0001049, 0.0001099
5: 0.0031032, 0.0036006, 0.0031240, 0.0036025, -0.0003960, 0.0003708
6: -0.0024373, -0.0023831, -0.0024361, -0.0023840, -0.0000533, 0.0000530
7: -0.0129425, -0.0121518, -0.0129428, -0.0122292, -0.0006989, 0.0007767
8: -0.0092612, -0.0076546, -0.0092693, -0.0076932, -0.0011442, 0.0011951
9: -0.0005610, 0.0002452, -0.0005458, 0.0002496, -0.0006000, 0.0005759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003319, upper bound: 0.0003286
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003319, upper bound: 0.0003341
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0010983, -0.0004542, -0.0010700, -0.0004511, -0.0005160, 0.0004848
1: -0.0042201, -0.0039964, -0.0042156, -0.0039949, -0.0001867, 0.0001848
2: 0.0131020, 0.0139794, 0.0131341, 0.0139842, -0.0006798, 0.0006446
3: 1.0084385, 1.0090141, 1.0084356, 1.0090029, -0.0005645, 0.0005656
4: -0.0038702, -0.0037238, -0.0038711, -0.0037282, -0.0001057, 0.0001100
5: 0.0031032, 0.0036007, 0.0031246, 0.0036031, -0.0003965, 0.0003729
6: -0.0024373, -0.0023831, -0.0024363, -0.0023841, -0.0000532, 0.0000531
7: -0.0129425, -0.0121516, -0.0129429, -0.0122226, -0.0007054, 0.0007770
8: -0.0092614, -0.0076550, -0.0092718, -0.0076964, -0.0011532, 0.0011962
9: -0.0005609, 0.0002454, -0.0005440, 0.0002509, -0.0006008, 0.0005805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003321, upper bound: 0.0003287
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003321, upper bound: 0.0003341
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010706, -0.0004518, -0.0010983, -0.0004542, -0.0004820, 0.0005153
1: -0.0042161, -0.0039955, -0.0042202, -0.0039965, -0.0001835, 0.0001865
2: 0.0131326, 0.0139830, 0.0131019, 0.0139793, -0.0006410, 0.0006789
3: 1.0084379, 1.0090041, 1.0084385, 1.0090144, -0.0005629, 0.0005656
4: -0.0038709, -0.0037278, -0.0038702, -0.0037237, -0.0001099, 0.0001049
5: 0.0031240, 0.0036025, 0.0031032, 0.0036006, -0.0003708, 0.0003960
6: -0.0024361, -0.0023840, -0.0024373, -0.0023831, -0.0000530, 0.0000533
7: -0.0129428, -0.0122292, -0.0129425, -0.0121518, -0.0007767, 0.0006989
8: -0.0092693, -0.0076932, -0.0092612, -0.0076546, -0.0011951, 0.0011442
9: -0.0005458, 0.0002496, -0.0005610, 0.0002452, -0.0005759, 0.0006000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003286, upper bound: 0.0003319
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003341, upper bound: 0.0003319
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010700, -0.0004511, -0.0010983, -0.0004542, -0.0004847, 0.0005160
1: -0.0042156, -0.0039949, -0.0042201, -0.0039964, -0.0001848, 0.0001866
2: 0.0131341, 0.0139842, 0.0131020, 0.0139794, -0.0006446, 0.0006798
3: 1.0084356, 1.0090029, 1.0084385, 1.0090141, -0.0005656, 0.0005645
4: -0.0038711, -0.0037282, -0.0038702, -0.0037238, -0.0001100, 0.0001057
5: 0.0031246, 0.0036031, 0.0031032, 0.0036007, -0.0003729, 0.0003965
6: -0.0024363, -0.0023841, -0.0024373, -0.0023831, -0.0000531, 0.0000532
7: -0.0129429, -0.0122226, -0.0129425, -0.0121516, -0.0007770, 0.0007054
8: -0.0092718, -0.0076964, -0.0092614, -0.0076550, -0.0011962, 0.0011532
9: -0.0005440, 0.0002509, -0.0005609, 0.0002454, -0.0005805, 0.0006008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003321
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003341, upper bound: 0.0003321
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010706, -0.0004518, -0.0010867, -0.0004492, -0.0004869, 0.0004996
1: -0.0042161, -0.0039955, -0.0042185, -0.0039941, -0.0001838, 0.0001828
2: 0.0131326, 0.0139830, 0.0131148, 0.0139871, -0.0006476, 0.0006594
3: 1.0084379, 1.0090041, 1.0084337, 1.0090100, -0.0005555, 0.0005657
4: -0.0038709, -0.0037278, -0.0038717, -0.0037255, -0.0001072, 0.0001063
5: 0.0031240, 0.0036025, 0.0031119, 0.0036046, -0.0003747, 0.0003840
6: -0.0024361, -0.0023840, -0.0024371, -0.0023835, -0.0000526, 0.0000532
7: -0.0129428, -0.0122292, -0.0129431, -0.0121838, -0.0007445, 0.0006998
8: -0.0092693, -0.0076932, -0.0092780, -0.0076710, -0.0011662, 0.0011592
9: -0.0005458, 0.0002496, -0.0005546, 0.0002542, -0.0005824, 0.0005857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003319
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003319
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010700, -0.0004511, -0.0010865, -0.0004491, -0.0004894, 0.0005003
1: -0.0042156, -0.0039949, -0.0042184, -0.0039941, -0.0001848, 0.0001831
2: 0.0131341, 0.0139842, 0.0131150, 0.0139872, -0.0006509, 0.0006600
3: 1.0084356, 1.0090029, 1.0084337, 1.0090098, -0.0005581, 0.0005684
4: -0.0038711, -0.0037282, -0.0038717, -0.0037255, -0.0001073, 0.0001069
5: 0.0031246, 0.0036031, 0.0031120, 0.0036046, -0.0003765, 0.0003845
6: -0.0024363, -0.0023841, -0.0024371, -0.0023835, -0.0000528, 0.0000531
7: -0.0129429, -0.0122226, -0.0129431, -0.0121834, -0.0007449, 0.0007062
8: -0.0092718, -0.0076964, -0.0092783, -0.0076713, -0.0011672, 0.0011663
9: -0.0005440, 0.0002509, -0.0005544, 0.0002544, -0.0005862, 0.0005864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003321
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003321
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0010929, -0.0004546, -0.0010975, -0.0004704, -0.0004786, 0.0005041
1: -0.0042195, -0.0039967, -0.0042245, -0.0040053, -0.0001630, 0.0001918
2: 0.0131077, 0.0139787, 0.0130971, 0.0139545, -0.0006235, 0.0006728
3: 1.0084395, 1.0090127, 1.0084658, 1.0090250, -0.0005854, 0.0005017
4: -0.0038701, -0.0037245, -0.0038656, -0.0037222, -0.0001104, 0.0000995
5: 0.0031072, 0.0036003, 0.0031033, 0.0035879, -0.0003671, 0.0003881
6: -0.0024368, -0.0023833, -0.0024333, -0.0023822, -0.0000545, 0.0000500
7: -0.0129425, -0.0121686, -0.0129406, -0.0122074, -0.0007208, 0.0007567
8: -0.0092598, -0.0076614, -0.0092075, -0.0076320, -0.0012045, 0.0010653
9: -0.0005586, 0.0002445, -0.0005766, 0.0002165, -0.0005231, 0.0006073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003066, upper bound: 0.0002943
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003099, upper bound: 0.0002966
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0010929, -0.0004546, -0.0010833, -0.0004651, -0.0004869, 0.0004896
1: -0.0042195, -0.0039967, -0.0042228, -0.0040032, -0.0001665, 0.0001910
2: 0.0131077, 0.0139787, 0.0131129, 0.0139626, -0.0006362, 0.0006573
3: 1.0084395, 1.0090127, 1.0084625, 1.0090209, -0.0005814, 0.0005034
4: -0.0038701, -0.0037245, -0.0038671, -0.0037241, -0.0001087, 0.0001019
5: 0.0031072, 0.0036003, 0.0031139, 0.0035921, -0.0003736, 0.0003770
6: -0.0024368, -0.0023833, -0.0024330, -0.0023826, -0.0000542, 0.0000498
7: -0.0129425, -0.0121686, -0.0129413, -0.0122432, -0.0006845, 0.0007577
8: -0.0092598, -0.0076614, -0.0092250, -0.0076480, -0.0011921, 0.0010930
9: -0.0005586, 0.0002445, -0.0005702, 0.0002259, -0.0005379, 0.0006044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003066, upper bound: 0.0002943
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003099, upper bound: 0.0002966
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004541, -0.0011097, -0.0004812, -0.0004915, 0.0005122
1: -0.0042203, -0.0039964, -0.0042254, -0.0040094, -0.0001770, 0.0001905
2: 0.0131006, 0.0139796, 0.0130838, 0.0139379, -0.0006417, 0.0006745
3: 1.0084382, 1.0090147, 1.0084695, 1.0090274, -0.0005891, 0.0005453
4: -0.0038703, -0.0037236, -0.0038625, -0.0037205, -0.0001097, 0.0001028
5: 0.0031023, 0.0036007, 0.0030942, 0.0035794, -0.0003772, 0.0003935
6: -0.0024374, -0.0023831, -0.0024351, -0.0023821, -0.0000553, 0.0000520
7: -0.0129426, -0.0121485, -0.0129394, -0.0121469, -0.0007816, 0.0007770
8: -0.0092618, -0.0076530, -0.0091715, -0.0076180, -0.0011924, 0.0011129
9: -0.0005617, 0.0002456, -0.0005802, 0.0001973, -0.0005561, 0.0005969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003088
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003095
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004541, -0.0010987, -0.0004762, -0.0005013, 0.0005035
1: -0.0042203, -0.0039964, -0.0042236, -0.0040070, -0.0001811, 0.0001898
2: 0.0131006, 0.0139796, 0.0130964, 0.0139456, -0.0006568, 0.0006651
3: 1.0084382, 1.0090147, 1.0084643, 1.0090227, -0.0005845, 0.0005498
4: -0.0038703, -0.0037236, -0.0038639, -0.0037222, -0.0001088, 0.0001057
5: 0.0031023, 0.0036007, 0.0031025, 0.0035833, -0.0003849, 0.0003869
6: -0.0024374, -0.0023831, -0.0024351, -0.0023824, -0.0000549, 0.0000520
7: -0.0129426, -0.0121485, -0.0129400, -0.0121725, -0.0007562, 0.0007782
8: -0.0092618, -0.0076530, -0.0091881, -0.0076340, -0.0011835, 0.0011457
9: -0.0005617, 0.0002456, -0.0005734, 0.0002062, -0.0005736, 0.0005940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003096
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003095
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010111, -0.0004433, -0.0011522, -0.0004797, -0.0003974, 0.0005630
1: -0.0042088, -0.0039928, -0.0042322, -0.0040084, -0.0001669, 0.0001860
2: 0.0131988, 0.0139961, 0.0130350, 0.0139402, -0.0005371, 0.0007362
3: 1.0084395, 1.0089859, 1.0084633, 1.0090444, -0.0005324, 0.0005226
4: -0.0038733, -0.0037362, -0.0038629, -0.0037139, -0.0001185, 0.0000898
5: 0.0031687, 0.0036092, 0.0030622, 0.0035806, -0.0003065, 0.0004321
6: -0.0024326, -0.0023855, -0.0024375, -0.0023807, -0.0000520, 0.0000520
7: -0.0129438, -0.0124041, -0.0129396, -0.0120467, -0.0008816, 0.0005216
8: -0.0092976, -0.0077673, -0.0091764, -0.0075563, -0.0012753, 0.0009922
9: -0.0005180, 0.0002647, -0.0006057, 0.0001999, -0.0005052, 0.0006279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003026
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003026
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010137, -0.0004429, -0.0011523, -0.0004796, -0.0004009, 0.0005655
1: -0.0042085, -0.0039923, -0.0042322, -0.0040084, -0.0001680, 0.0001866
2: 0.0131964, 0.0139968, 0.0130349, 0.0139403, -0.0005414, 0.0007399
3: 1.0084368, 1.0089853, 1.0084633, 1.0090443, -0.0005357, 0.0005220
4: -0.0038735, -0.0037361, -0.0038630, -0.0037139, -0.0001192, 0.0000905
5: 0.0031668, 0.0036096, 0.0030621, 0.0035806, -0.0003091, 0.0004341
6: -0.0024329, -0.0023855, -0.0024375, -0.0023807, -0.0000523, 0.0000520
7: -0.0129439, -0.0123941, -0.0129396, -0.0120462, -0.0008824, 0.0005317
8: -0.0092991, -0.0077684, -0.0091767, -0.0075561, -0.0012837, 0.0009988
9: -0.0005170, 0.0002655, -0.0006056, 0.0002001, -0.0005087, 0.0006321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003036, upper bound: 0.0003046
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003036, upper bound: 0.0003046
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010300, -0.0004531, -0.0011593, -0.0004791, -0.0004144, 0.0005793
1: -0.0042096, -0.0039965, -0.0042331, -0.0040081, -0.0001669, 0.0001995
2: 0.0131791, 0.0139810, 0.0130272, 0.0139411, -0.0005501, 0.0007592
3: 1.0084426, 1.0089878, 1.0084616, 1.0090464, -0.0005762, 0.0005262
4: -0.0038705, -0.0037341, -0.0038631, -0.0037129, -0.0001228, 0.0000902
5: 0.0031546, 0.0036015, 0.0030568, 0.0035810, -0.0003188, 0.0004447
6: -0.0024341, -0.0023853, -0.0024382, -0.0023805, -0.0000536, 0.0000529
7: -0.0129427, -0.0123246, -0.0129396, -0.0120266, -0.0009028, 0.0006018
8: -0.0092650, -0.0077523, -0.0091784, -0.0075476, -0.0013282, 0.0009879
9: -0.0005213, 0.0002472, -0.0006089, 0.0002010, -0.0004985, 0.0006600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002915
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002992
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010284, -0.0004524, -0.0011594, -0.0004790, -0.0004147, 0.0005797
1: -0.0042089, -0.0039960, -0.0042331, -0.0040080, -0.0001673, 0.0001994
2: 0.0131819, 0.0139821, 0.0130271, 0.0139412, -0.0005508, 0.0007600
3: 1.0084403, 1.0089861, 1.0084617, 1.0090463, -0.0005777, 0.0005244
4: -0.0038707, -0.0037347, -0.0038631, -0.0037129, -0.0001230, 0.0000904
5: 0.0031559, 0.0036020, 0.0030568, 0.0035811, -0.0003189, 0.0004450
6: -0.0024342, -0.0023855, -0.0024382, -0.0023805, -0.0000537, 0.0000528
7: -0.0129427, -0.0123237, -0.0129396, -0.0120261, -0.0009033, 0.0006026
8: -0.0092672, -0.0077580, -0.0091786, -0.0075475, -0.0013295, 0.0009903
9: -0.0005188, 0.0002484, -0.0006088, 0.0002011, -0.0004997, 0.0006606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002930
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002996
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010975, -0.0004704, -0.0010929, -0.0004546, -0.0005041, 0.0004786
1: -0.0042245, -0.0040053, -0.0042195, -0.0039967, -0.0001918, 0.0001630
2: 0.0130971, 0.0139545, 0.0131077, 0.0139787, -0.0006728, 0.0006235
3: 1.0084658, 1.0090250, 1.0084395, 1.0090127, -0.0005017, 0.0005854
4: -0.0038656, -0.0037222, -0.0038701, -0.0037245, -0.0000995, 0.0001104
5: 0.0031033, 0.0035879, 0.0031072, 0.0036003, -0.0003881, 0.0003671
6: -0.0024333, -0.0023822, -0.0024368, -0.0023833, -0.0000500, 0.0000545
7: -0.0129406, -0.0122074, -0.0129425, -0.0121686, -0.0007567, 0.0007208
8: -0.0092075, -0.0076320, -0.0092598, -0.0076614, -0.0010653, 0.0012045
9: -0.0005766, 0.0002165, -0.0005586, 0.0002445, -0.0006073, 0.0005231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002943, upper bound: 0.0003066
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003098
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010833, -0.0004651, -0.0010929, -0.0004546, -0.0004896, 0.0004869
1: -0.0042228, -0.0040032, -0.0042195, -0.0039967, -0.0001910, 0.0001665
2: 0.0131129, 0.0139626, 0.0131077, 0.0139787, -0.0006573, 0.0006362
3: 1.0084625, 1.0090209, 1.0084395, 1.0090127, -0.0005034, 0.0005814
4: -0.0038671, -0.0037241, -0.0038701, -0.0037245, -0.0001019, 0.0001087
5: 0.0031139, 0.0035921, 0.0031072, 0.0036003, -0.0003770, 0.0003736
6: -0.0024330, -0.0023826, -0.0024368, -0.0023833, -0.0000498, 0.0000542
7: -0.0129413, -0.0122432, -0.0129425, -0.0121686, -0.0007577, 0.0006845
8: -0.0092250, -0.0076480, -0.0092598, -0.0076614, -0.0010930, 0.0011921
9: -0.0005702, 0.0002259, -0.0005586, 0.0002445, -0.0006044, 0.0005379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002943, upper bound: 0.0003066
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003099
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0011097, -0.0004812, -0.0010995, -0.0004541, -0.0005122, 0.0004915
1: -0.0042254, -0.0040094, -0.0042203, -0.0039964, -0.0001905, 0.0001770
2: 0.0130838, 0.0139379, 0.0131006, 0.0139796, -0.0006745, 0.0006417
3: 1.0084695, 1.0090274, 1.0084382, 1.0090147, -0.0005453, 0.0005891
4: -0.0038625, -0.0037205, -0.0038703, -0.0037236, -0.0001028, 0.0001097
5: 0.0030942, 0.0035794, 0.0031023, 0.0036007, -0.0003935, 0.0003772
6: -0.0024351, -0.0023821, -0.0024374, -0.0023831, -0.0000520, 0.0000553
7: -0.0129394, -0.0121469, -0.0129426, -0.0121485, -0.0007770, 0.0007816
8: -0.0091715, -0.0076180, -0.0092618, -0.0076530, -0.0011129, 0.0011924
9: -0.0005802, 0.0001973, -0.0005617, 0.0002456, -0.0005969, 0.0005561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003095, upper bound: 0.0003118
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010987, -0.0004762, -0.0010995, -0.0004541, -0.0005035, 0.0005013
1: -0.0042236, -0.0040070, -0.0042203, -0.0039964, -0.0001898, 0.0001811
2: 0.0130964, 0.0139456, 0.0131006, 0.0139796, -0.0006651, 0.0006568
3: 1.0084643, 1.0090227, 1.0084382, 1.0090147, -0.0005498, 0.0005845
4: -0.0038639, -0.0037222, -0.0038703, -0.0037236, -0.0001057, 0.0001088
5: 0.0031025, 0.0035833, 0.0031023, 0.0036007, -0.0003869, 0.0003849
6: -0.0024351, -0.0023824, -0.0024374, -0.0023831, -0.0000520, 0.0000549
7: -0.0129400, -0.0121725, -0.0129426, -0.0121485, -0.0007782, 0.0007562
8: -0.0091881, -0.0076340, -0.0092618, -0.0076530, -0.0011457, 0.0011835
9: -0.0005734, 0.0002062, -0.0005617, 0.0002456, -0.0005940, 0.0005736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0003118
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003095, upper bound: 0.0003118
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011522, -0.0004797, -0.0010111, -0.0004433, -0.0005630, 0.0003974
1: -0.0042322, -0.0040084, -0.0042088, -0.0039928, -0.0001860, 0.0001669
2: 0.0130350, 0.0139402, 0.0131988, 0.0139961, -0.0007362, 0.0005371
3: 1.0084633, 1.0090444, 1.0084395, 1.0089859, -0.0005226, 0.0005324
4: -0.0038629, -0.0037139, -0.0038733, -0.0037362, -0.0000898, 0.0001185
5: 0.0030622, 0.0035806, 0.0031687, 0.0036092, -0.0004321, 0.0003065
6: -0.0024375, -0.0023807, -0.0024326, -0.0023855, -0.0000520, 0.0000520
7: -0.0129396, -0.0120467, -0.0129438, -0.0124041, -0.0005216, 0.0008816
8: -0.0091764, -0.0075563, -0.0092976, -0.0077673, -0.0009922, 0.0012753
9: -0.0006057, 0.0001999, -0.0005180, 0.0002647, -0.0006279, 0.0005052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011523, -0.0004796, -0.0010137, -0.0004429, -0.0005655, 0.0004009
1: -0.0042322, -0.0040084, -0.0042085, -0.0039923, -0.0001866, 0.0001680
2: 0.0130349, 0.0139403, 0.0131964, 0.0139968, -0.0007399, 0.0005414
3: 1.0084633, 1.0090443, 1.0084368, 1.0089853, -0.0005220, 0.0005357
4: -0.0038630, -0.0037139, -0.0038735, -0.0037361, -0.0000905, 0.0001192
5: 0.0030621, 0.0035806, 0.0031668, 0.0036096, -0.0004341, 0.0003091
6: -0.0024375, -0.0023807, -0.0024329, -0.0023855, -0.0000520, 0.0000523
7: -0.0129396, -0.0120462, -0.0129439, -0.0123941, -0.0005317, 0.0008824
8: -0.0091767, -0.0075561, -0.0092991, -0.0077684, -0.0009988, 0.0012837
9: -0.0006056, 0.0002001, -0.0005170, 0.0002655, -0.0006321, 0.0005087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003046, upper bound: 0.0003036
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003046, upper bound: 0.0003036
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011593, -0.0004791, -0.0010300, -0.0004531, -0.0005793, 0.0004144
1: -0.0042331, -0.0040081, -0.0042096, -0.0039965, -0.0001995, 0.0001669
2: 0.0130272, 0.0139411, 0.0131791, 0.0139810, -0.0007592, 0.0005501
3: 1.0084616, 1.0090464, 1.0084426, 1.0089878, -0.0005262, 0.0005762
4: -0.0038631, -0.0037129, -0.0038705, -0.0037341, -0.0000902, 0.0001228
5: 0.0030568, 0.0035810, 0.0031546, 0.0036015, -0.0004447, 0.0003188
6: -0.0024382, -0.0023805, -0.0024341, -0.0023853, -0.0000529, 0.0000536
7: -0.0129396, -0.0120266, -0.0129427, -0.0123246, -0.0006018, 0.0009028
8: -0.0091784, -0.0075476, -0.0092650, -0.0077523, -0.0009879, 0.0013282
9: -0.0006089, 0.0002010, -0.0005213, 0.0002472, -0.0006600, 0.0004985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003085
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003079
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011594, -0.0004790, -0.0010284, -0.0004524, -0.0005797, 0.0004147
1: -0.0042331, -0.0040080, -0.0042089, -0.0039960, -0.0001994, 0.0001673
2: 0.0130271, 0.0139412, 0.0131819, 0.0139821, -0.0007600, 0.0005508
3: 1.0084617, 1.0090463, 1.0084403, 1.0089861, -0.0005244, 0.0005777
4: -0.0038631, -0.0037129, -0.0038707, -0.0037347, -0.0000904, 0.0001230
5: 0.0030568, 0.0035811, 0.0031559, 0.0036020, -0.0004450, 0.0003189
6: -0.0024382, -0.0023805, -0.0024342, -0.0023855, -0.0000528, 0.0000537
7: -0.0129396, -0.0120261, -0.0129427, -0.0123237, -0.0006026, 0.0009033
8: -0.0091786, -0.0075475, -0.0092672, -0.0077580, -0.0009903, 0.0013295
9: -0.0006088, 0.0002011, -0.0005188, 0.0002484, -0.0006606, 0.0004997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003096
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003080
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010771, -0.0004732, -0.0011554, -0.0004794, -0.0004530, 0.0005249
1: -0.0042220, -0.0040068, -0.0042327, -0.0040083, -0.0001774, 0.0001705
2: 0.0131204, 0.0139501, 0.0130315, 0.0139406, -0.0005986, 0.0006741
3: 1.0084703, 1.0090185, 1.0084629, 1.0090455, -0.0005223, 0.0005556
4: -0.0038648, -0.0037251, -0.0038630, -0.0037134, -0.0001060, 0.0000974
5: 0.0031187, 0.0035857, 0.0030598, 0.0035808, -0.0003483, 0.0004019
6: -0.0024322, -0.0023828, -0.0024377, -0.0023806, -0.0000516, 0.0000550
7: -0.0129403, -0.0122588, -0.0129396, -0.0120388, -0.0008848, 0.0006655
8: -0.0091979, -0.0076567, -0.0091772, -0.0075520, -0.0011244, 0.0010631
9: -0.0005670, 0.0002114, -0.0006075, 0.0002004, -0.0005353, 0.0005473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002772, upper bound: 0.0002921
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002772, upper bound: 0.0002921
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010823, -0.0004714, -0.0011554, -0.0004794, -0.0004564, 0.0005281
1: -0.0042217, -0.0040061, -0.0042327, -0.0040082, -0.0001780, 0.0001715
2: 0.0131146, 0.0139530, 0.0130314, 0.0139407, -0.0006023, 0.0006790
3: 1.0084664, 1.0090178, 1.0084625, 1.0090454, -0.0005262, 0.0005553
4: -0.0038653, -0.0037245, -0.0038630, -0.0037134, -0.0001069, 0.0000978
5: 0.0031148, 0.0035871, 0.0030597, 0.0035808, -0.0003507, 0.0004044
6: -0.0024326, -0.0023828, -0.0024377, -0.0023806, -0.0000521, 0.0000549
7: -0.0129405, -0.0122452, -0.0129396, -0.0120383, -0.0008857, 0.0006795
8: -0.0092041, -0.0076551, -0.0091774, -0.0075518, -0.0011347, 0.0010669
9: -0.0005659, 0.0002147, -0.0006074, 0.0002005, -0.0005370, 0.0005528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002792, upper bound: 0.0002945
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002792, upper bound: 0.0002945
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010633, -0.0004680, -0.0011522, -0.0004797, -0.0004412, 0.0005345
1: -0.0042203, -0.0040047, -0.0042322, -0.0040084, -0.0001766, 0.0001739
2: 0.0131349, 0.0139582, 0.0130350, 0.0139402, -0.0005872, 0.0006903
3: 1.0084674, 1.0090146, 1.0084633, 1.0090444, -0.0005229, 0.0005513
4: -0.0038663, -0.0037268, -0.0038629, -0.0037139, -0.0001092, 0.0000963
5: 0.0031290, 0.0035898, 0.0030622, 0.0035806, -0.0003395, 0.0004096
6: -0.0024320, -0.0023831, -0.0024375, -0.0023807, -0.0000513, 0.0000544
7: -0.0129409, -0.0122942, -0.0129396, -0.0120467, -0.0008784, 0.0006304
8: -0.0092154, -0.0076729, -0.0091764, -0.0075563, -0.0011628, 0.0010551
9: -0.0005605, 0.0002208, -0.0006057, 0.0001999, -0.0005329, 0.0005686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002722, upper bound: 0.0002844
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002722, upper bound: 0.0002844
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010678, -0.0004667, -0.0011523, -0.0004796, -0.0004434, 0.0005372
1: -0.0042200, -0.0040040, -0.0042322, -0.0040084, -0.0001769, 0.0001746
2: 0.0131306, 0.0139602, 0.0130349, 0.0139403, -0.0005896, 0.0006942
3: 1.0084640, 1.0090137, 1.0084633, 1.0090443, -0.0005261, 0.0005504
4: -0.0038667, -0.0037266, -0.0038630, -0.0037139, -0.0001099, 0.0000966
5: 0.0031256, 0.0035908, 0.0030621, 0.0035806, -0.0003412, 0.0004117
6: -0.0024324, -0.0023832, -0.0024375, -0.0023807, -0.0000517, 0.0000544
7: -0.0129411, -0.0122834, -0.0129396, -0.0120462, -0.0008792, 0.0006411
8: -0.0092198, -0.0076729, -0.0091767, -0.0075561, -0.0011714, 0.0010572
9: -0.0005596, 0.0002231, -0.0006056, 0.0002001, -0.0005339, 0.0005732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002878
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002878
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011097, -0.0004812, -0.0011083, -0.0004693, -0.0004936, 0.0004820
1: -0.0042254, -0.0040094, -0.0042263, -0.0040047, -0.0001707, 0.0001790
2: 0.0130838, 0.0139379, 0.0130850, 0.0139562, -0.0006424, 0.0006315
3: 1.0084695, 1.0090274, 1.0084635, 1.0090295, -0.0005600, 0.0005359
4: -0.0038625, -0.0037205, -0.0038659, -0.0037205, -0.0001019, 0.0001024
5: 0.0030942, 0.0035794, 0.0030952, 0.0035888, -0.0003786, 0.0003699
6: -0.0024351, -0.0023821, -0.0024340, -0.0023819, -0.0000532, 0.0000519
7: -0.0129394, -0.0121469, -0.0129408, -0.0121795, -0.0007452, 0.0007796
8: -0.0091715, -0.0076180, -0.0092111, -0.0076158, -0.0011068, 0.0010944
9: -0.0005802, 0.0001973, -0.0005833, 0.0002185, -0.0005369, 0.0005559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002812
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002812
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011097, -0.0004812, -0.0011213, -0.0004799, -0.0004809, 0.0004927
1: -0.0042254, -0.0040094, -0.0042272, -0.0040087, -0.0001766, 0.0001776
2: 0.0130838, 0.0139379, 0.0130706, 0.0139399, -0.0006253, 0.0006374
3: 1.0084695, 1.0090274, 1.0084666, 1.0090318, -0.0005623, 0.0005608
4: -0.0038625, -0.0037205, -0.0038629, -0.0037187, -0.0001012, 0.0000995
5: 0.0030942, 0.0035794, 0.0030854, 0.0035804, -0.0003688, 0.0003776
6: -0.0024351, -0.0023821, -0.0024360, -0.0023817, -0.0000534, 0.0000539
7: -0.0129394, -0.0121469, -0.0129395, -0.0121160, -0.0008095, 0.0007782
8: -0.0091715, -0.0076180, -0.0091757, -0.0076015, -0.0010918, 0.0010758
9: -0.0005802, 0.0001973, -0.0005868, 0.0001995, -0.0005380, 0.0005448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002911
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002911
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010816, -0.0004789, -0.0011593, -0.0004791, -0.0004566, 0.0005490
1: -0.0042211, -0.0040085, -0.0042331, -0.0040081, -0.0001757, 0.0001885
2: 0.0131165, 0.0139413, 0.0130272, 0.0139411, -0.0005981, 0.0007104
3: 1.0084692, 1.0090163, 1.0084616, 1.0090464, -0.0005690, 0.0005547
4: -0.0038631, -0.0037249, -0.0038631, -0.0037129, -0.0001130, 0.0000962
5: 0.0031154, 0.0035812, 0.0030568, 0.0035810, -0.0003506, 0.0004207
6: -0.0024340, -0.0023830, -0.0024382, -0.0023805, -0.0000535, 0.0000553
7: -0.0129396, -0.0122152, -0.0129396, -0.0120266, -0.0008994, 0.0007102
8: -0.0091789, -0.0076594, -0.0091784, -0.0075476, -0.0012140, 0.0010452
9: -0.0005638, 0.0002013, -0.0006089, 0.0002010, -0.0005232, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002725
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002871
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010814, -0.0004769, -0.0011594, -0.0004790, -0.0004558, 0.0005494
1: -0.0042208, -0.0040078, -0.0042331, -0.0040080, -0.0001759, 0.0001884
2: 0.0131161, 0.0139445, 0.0130271, 0.0139412, -0.0005973, 0.0007110
3: 1.0084671, 1.0090158, 1.0084617, 1.0090463, -0.0005701, 0.0005541
4: -0.0038637, -0.0037248, -0.0038631, -0.0037129, -0.0001131, 0.0000962
5: 0.0031155, 0.0035828, 0.0030568, 0.0035811, -0.0003499, 0.0004211
6: -0.0024340, -0.0023830, -0.0024382, -0.0023805, -0.0000535, 0.0000552
7: -0.0129399, -0.0122171, -0.0129396, -0.0120261, -0.0008999, 0.0007087
8: -0.0091858, -0.0076589, -0.0091786, -0.0075475, -0.0012150, 0.0010459
9: -0.0005629, 0.0002050, -0.0006088, 0.0002011, -0.0005236, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002761
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002873
time: 0.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.66 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002487, upper bound: 0.0002487
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002487, upper bound: 0.0002987
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0002712
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0003351
IS_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003319, upper bound: 0.0003286
IS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003319, upper bound: 0.0003341
IS_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003321, upper bound: 0.0003287
IS_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003321, upper bound: 0.0003341
IS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003286, upper bound: 0.0003319
IS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003341, upper bound: 0.0003319
IS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003321
IS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003341, upper bound: 0.0003321
IS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003319
IS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003319
IS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003321
IS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003321
IS_A1_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003066, upper bound: 0.0002943
IS_A1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003099, upper bound: 0.0002966
IS_A1_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003066, upper bound: 0.0002943
IS_A1_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003099, upper bound: 0.0002966
IS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003088
IS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003095
IS_A1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003096
IS_A1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003095
IS_A1_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003026
IS_A1_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003026
IS_A1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003036, upper bound: 0.0003046
IS_A1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003036, upper bound: 0.0003046
IS_A1_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002915
IS_A1_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002992
IS_A1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002930
IS_A1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002996
IS_A2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002943, upper bound: 0.0003066
IS_A2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003098
IS_A2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002943, upper bound: 0.0003066
IS_A2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002966, upper bound: 0.0003099
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003095, upper bound: 0.0003118
IS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0003118
IS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003095, upper bound: 0.0003118
IS_A2_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
IS_A2_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
IS_A2_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003046, upper bound: 0.0003036
IS_A2_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0003046, upper bound: 0.0003036
IS_A2_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003085
IS_A2_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003079
IS_A2_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003096
IS_A2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003080
IS_A2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002772, upper bound: 0.0002921
IS_A2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002772, upper bound: 0.0002921
IS_A2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002792, upper bound: 0.0002945
IS_A2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002792, upper bound: 0.0002945
IS_A2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002722, upper bound: 0.0002844
IS_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002722, upper bound: 0.0002844
IS_A2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002878
IS_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002878
IS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002812
IS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002812
IS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002911
IS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002895, upper bound: 0.0002911
IS_A2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002725
IS_A2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002871
IS_A2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002761
IS_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.66
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002873

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011323, -0.0005051, -0.0010995, -0.0004549, -0.0005410, 0.0004656
1: -0.0042285, -0.0040199, -0.0042203, -0.0039968, -0.0001966, 0.0001637
2: 0.0130606, 0.0139011, 0.0131006, 0.0139783, -0.0007082, 0.0006003
3: 1.0084885, 1.0090350, 1.0084393, 1.0090147, -0.0005262, 0.0005957
4: -0.0038556, -0.0037176, -0.0038700, -0.0037236, -0.0000948, 0.0001146
5: 0.0030775, 0.0035605, 0.0031023, 0.0036001, -0.0004151, 0.0003567
6: -0.0024356, -0.0023814, -0.0024373, -0.0023831, -0.0000525, 0.0000559
7: -0.0129366, -0.0120880, -0.0129425, -0.0121485, -0.0007743, 0.0008393
8: -0.0090917, -0.0075900, -0.0092589, -0.0076530, -0.0010063, 0.0012426
9: -0.0005918, 0.0001546, -0.0005617, 0.0002440, -0.0006241, 0.0004899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002482, upper bound: 0.0002987
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002480, upper bound: 0.0002986
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004549, -0.0011323, -0.0005051, -0.0004656, 0.0005410
1: -0.0042203, -0.0039968, -0.0042285, -0.0040199, -0.0001637, 0.0001966
2: 0.0131006, 0.0139783, 0.0130606, 0.0139011, -0.0006003, 0.0007082
3: 1.0084393, 1.0090147, 1.0084885, 1.0090350, -0.0005957, 0.0005262
4: -0.0038700, -0.0037236, -0.0038556, -0.0037176, -0.0001146, 0.0000948
5: 0.0031023, 0.0036001, 0.0030775, 0.0035605, -0.0003567, 0.0004151
6: -0.0024373, -0.0023831, -0.0024356, -0.0023814, -0.0000559, 0.0000525
7: -0.0129425, -0.0121485, -0.0129366, -0.0120880, -0.0008393, 0.0007743
8: -0.0092589, -0.0076530, -0.0090917, -0.0075900, -0.0012426, 0.0010063
9: -0.0005617, 0.0002440, -0.0005918, 0.0001546, -0.0004899, 0.0006241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0002711
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002986, upper bound: 0.0002711
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010995, -0.0004549, -0.0010995, -0.0004549, -0.0005077, 0.0005077
1: -0.0042203, -0.0039968, -0.0042203, -0.0039968, -0.0001907, 0.0001907
2: 0.0131006, 0.0139783, 0.0131006, 0.0139783, -0.0006673, 0.0006673
3: 1.0084393, 1.0090147, 1.0084393, 1.0090147, -0.0005754, 0.0005754
4: -0.0038700, -0.0037236, -0.0038700, -0.0037236, -0.0001088, 0.0001088
5: 0.0031023, 0.0036001, 0.0031023, 0.0036001, -0.0003898, 0.0003898
6: -0.0024373, -0.0023831, -0.0024373, -0.0023831, -0.0000542, 0.0000542
7: -0.0129425, -0.0121485, -0.0129425, -0.0121485, -0.0007790, 0.0007790
8: -0.0092589, -0.0076530, -0.0092589, -0.0076530, -0.0011863, 0.0011863
9: -0.0005617, 0.0002440, -0.0005617, 0.0002440, -0.0005986, 0.0005986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0003244
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002986, upper bound: 0.0003249
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0010917, -0.0004548, -0.0010111, -0.0004433, -0.0004993, 0.0004120
1: -0.0042194, -0.0039968, -0.0042088, -0.0039928, -0.0001680, 0.0001726
2: 0.0131091, 0.0139784, 0.0131988, 0.0139961, -0.0006560, 0.0005614
3: 1.0084399, 1.0090123, 1.0084395, 1.0089859, -0.0005411, 0.0004877
4: -0.0038700, -0.0037247, -0.0038733, -0.0037362, -0.0000945, 0.0001057
5: 0.0031082, 0.0036001, 0.0031687, 0.0036092, -0.0003835, 0.0003180
6: -0.0024367, -0.0023833, -0.0024326, -0.0023855, -0.0000512, 0.0000493
7: -0.0129425, -0.0121718, -0.0129438, -0.0124041, -0.0005233, 0.0007560
8: -0.0092592, -0.0076630, -0.0092976, -0.0077673, -0.0010498, 0.0011372
9: -0.0005580, 0.0002442, -0.0005180, 0.0002647, -0.0005618, 0.0005372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002913, upper bound: 0.0002685
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003311, upper bound: 0.0003278
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0010983, -0.0004542, -0.0010300, -0.0004531, -0.0005146, 0.0004296
1: -0.0042202, -0.0039965, -0.0042096, -0.0039965, -0.0001829, 0.0001716
2: 0.0131019, 0.0139793, 0.0131791, 0.0139810, -0.0006778, 0.0005748
3: 1.0084385, 1.0090144, 1.0084426, 1.0089878, -0.0005451, 0.0005349
4: -0.0038702, -0.0037237, -0.0038705, -0.0037341, -0.0000947, 0.0001097
5: 0.0031032, 0.0036006, 0.0031546, 0.0036015, -0.0003954, 0.0003307
6: -0.0024373, -0.0023831, -0.0024341, -0.0023853, -0.0000520, 0.0000509
7: -0.0129425, -0.0121518, -0.0129427, -0.0123246, -0.0006035, 0.0007766
8: -0.0092612, -0.0076546, -0.0092650, -0.0077523, -0.0010407, 0.0011927
9: -0.0005610, 0.0002452, -0.0005213, 0.0002472, -0.0005987, 0.0005277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002913, upper bound: 0.0002690
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003311, upper bound: 0.0003333
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0010917, -0.0004548, -0.0010137, -0.0004429, -0.0005020, 0.0004159
1: -0.0042193, -0.0039968, -0.0042085, -0.0039923, -0.0001686, 0.0001739
2: 0.0131091, 0.0139785, 0.0131964, 0.0139968, -0.0006598, 0.0005663
3: 1.0084395, 1.0090122, 1.0084368, 1.0089853, -0.0005443, 0.0004910
4: -0.0038701, -0.0037247, -0.0038735, -0.0037361, -0.0000953, 0.0001064
5: 0.0031082, 0.0036002, 0.0031668, 0.0036096, -0.0003855, 0.0003211
6: -0.0024367, -0.0023833, -0.0024329, -0.0023855, -0.0000512, 0.0000496
7: -0.0129425, -0.0121716, -0.0129439, -0.0123941, -0.0005333, 0.0007565
8: -0.0092595, -0.0076634, -0.0092991, -0.0077684, -0.0010586, 0.0011453
9: -0.0005578, 0.0002443, -0.0005170, 0.0002655, -0.0005661, 0.0005417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002914, upper bound: 0.0002687
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003313, upper bound: 0.0003278
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0010983, -0.0004542, -0.0010284, -0.0004524, -0.0005152, 0.0004299
1: -0.0042201, -0.0039964, -0.0042089, -0.0039960, -0.0001828, 0.0001721
2: 0.0131020, 0.0139794, 0.0131819, 0.0139821, -0.0006785, 0.0005751
3: 1.0084385, 1.0090141, 1.0084403, 1.0089861, -0.0005461, 0.0005364
4: -0.0038702, -0.0037238, -0.0038707, -0.0037347, -0.0000949, 0.0001098
5: 0.0031032, 0.0036007, 0.0031559, 0.0036020, -0.0003958, 0.0003309
6: -0.0024373, -0.0023831, -0.0024342, -0.0023855, -0.0000518, 0.0000510
7: -0.0129425, -0.0121516, -0.0129427, -0.0123237, -0.0006041, 0.0007769
8: -0.0092614, -0.0076550, -0.0092672, -0.0077580, -0.0010433, 0.0011935
9: -0.0005609, 0.0002454, -0.0005188, 0.0002484, -0.0005994, 0.0005290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002914, upper bound: 0.0002690
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003313, upper bound: 0.0003333
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010111, -0.0004433, -0.0010917, -0.0004548, -0.0004120, 0.0004993
1: -0.0042088, -0.0039928, -0.0042194, -0.0039968, -0.0001726, 0.0001680
2: 0.0131988, 0.0139961, 0.0131091, 0.0139784, -0.0005614, 0.0006560
3: 1.0084395, 1.0089859, 1.0084399, 1.0090123, -0.0004877, 0.0005411
4: -0.0038733, -0.0037362, -0.0038700, -0.0037247, -0.0001057, 0.0000945
5: 0.0031687, 0.0036092, 0.0031082, 0.0036001, -0.0003180, 0.0003835
6: -0.0024326, -0.0023855, -0.0024367, -0.0023833, -0.0000493, 0.0000512
7: -0.0129438, -0.0124041, -0.0129425, -0.0121718, -0.0007560, 0.0005233
8: -0.0092976, -0.0077673, -0.0092592, -0.0076630, -0.0011372, 0.0010498
9: -0.0005180, 0.0002647, -0.0005580, 0.0002442, -0.0005372, 0.0005618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002685, upper bound: 0.0002913
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003277, upper bound: 0.0003311
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010300, -0.0004531, -0.0010983, -0.0004542, -0.0004296, 0.0005146
1: -0.0042096, -0.0039965, -0.0042202, -0.0039965, -0.0001716, 0.0001829
2: 0.0131791, 0.0139810, 0.0131019, 0.0139793, -0.0005748, 0.0006778
3: 1.0084426, 1.0089878, 1.0084385, 1.0090144, -0.0005349, 0.0005451
4: -0.0038705, -0.0037341, -0.0038702, -0.0037237, -0.0001097, 0.0000947
5: 0.0031546, 0.0036015, 0.0031032, 0.0036006, -0.0003307, 0.0003954
6: -0.0024341, -0.0023853, -0.0024373, -0.0023831, -0.0000509, 0.0000520
7: -0.0129427, -0.0123246, -0.0129425, -0.0121518, -0.0007766, 0.0006035
8: -0.0092650, -0.0077523, -0.0092612, -0.0076546, -0.0011927, 0.0010407
9: -0.0005213, 0.0002472, -0.0005610, 0.0002452, -0.0005277, 0.0005987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002690, upper bound: 0.0002913
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003333, upper bound: 0.0003311
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010137, -0.0004429, -0.0010917, -0.0004548, -0.0004159, 0.0005020
1: -0.0042085, -0.0039923, -0.0042193, -0.0039968, -0.0001739, 0.0001686
2: 0.0131964, 0.0139968, 0.0131091, 0.0139785, -0.0005663, 0.0006598
3: 1.0084368, 1.0089853, 1.0084395, 1.0090122, -0.0004910, 0.0005443
4: -0.0038735, -0.0037361, -0.0038701, -0.0037247, -0.0001064, 0.0000953
5: 0.0031668, 0.0036096, 0.0031082, 0.0036002, -0.0003211, 0.0003855
6: -0.0024329, -0.0023855, -0.0024367, -0.0023833, -0.0000496, 0.0000512
7: -0.0129439, -0.0123941, -0.0129425, -0.0121716, -0.0007565, 0.0005333
8: -0.0092991, -0.0077684, -0.0092595, -0.0076634, -0.0011453, 0.0010586
9: -0.0005170, 0.0002655, -0.0005578, 0.0002443, -0.0005417, 0.0005661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002687, upper bound: 0.0002914
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003278, upper bound: 0.0003313
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010284, -0.0004524, -0.0010983, -0.0004542, -0.0004299, 0.0005152
1: -0.0042089, -0.0039960, -0.0042201, -0.0039964, -0.0001721, 0.0001828
2: 0.0131819, 0.0139821, 0.0131020, 0.0139794, -0.0005751, 0.0006785
3: 1.0084403, 1.0089861, 1.0084385, 1.0090141, -0.0005364, 0.0005461
4: -0.0038707, -0.0037347, -0.0038702, -0.0037238, -0.0001098, 0.0000949
5: 0.0031559, 0.0036020, 0.0031032, 0.0036007, -0.0003309, 0.0003958
6: -0.0024342, -0.0023855, -0.0024373, -0.0023831, -0.0000510, 0.0000518
7: -0.0129427, -0.0123237, -0.0129425, -0.0121516, -0.0007769, 0.0006041
8: -0.0092672, -0.0077580, -0.0092614, -0.0076550, -0.0011935, 0.0010433
9: -0.0005188, 0.0002484, -0.0005609, 0.0002454, -0.0005290, 0.0005994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002690, upper bound: 0.0002914
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003333, upper bound: 0.0003313
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010111, -0.0004433, -0.0010795, -0.0004498, -0.0004193, 0.0004825
1: -0.0042088, -0.0039928, -0.0042176, -0.0039944, -0.0001730, 0.0001644
2: 0.0131988, 0.0139961, 0.0131224, 0.0139862, -0.0005708, 0.0006349
3: 1.0084395, 1.0089859, 1.0084351, 1.0090079, -0.0004798, 0.0005358
4: -0.0038733, -0.0037362, -0.0038715, -0.0037264, -0.0001025, 0.0000961
5: 0.0031687, 0.0036092, 0.0031173, 0.0036041, -0.0003237, 0.0003707
6: -0.0024326, -0.0023855, -0.0024366, -0.0023837, -0.0000490, 0.0000511
7: -0.0129438, -0.0124041, -0.0129431, -0.0122040, -0.0007234, 0.0005242
8: -0.0092976, -0.0077673, -0.0092760, -0.0076792, -0.0011040, 0.0010652
9: -0.0005180, 0.0002647, -0.0005515, 0.0002532, -0.0005442, 0.0005459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003270
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003319
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010300, -0.0004531, -0.0010867, -0.0004492, -0.0004346, 0.0004989
1: -0.0042096, -0.0039965, -0.0042185, -0.0039941, -0.0001719, 0.0001792
2: 0.0131791, 0.0139810, 0.0131148, 0.0139871, -0.0005811, 0.0006582
3: 1.0084426, 1.0089878, 1.0084337, 1.0090100, -0.0005273, 0.0005390
4: -0.0038705, -0.0037341, -0.0038717, -0.0037255, -0.0001070, 0.0000960
5: 0.0031546, 0.0036015, 0.0031119, 0.0036046, -0.0003347, 0.0003834
6: -0.0024341, -0.0023853, -0.0024371, -0.0023835, -0.0000506, 0.0000518
7: -0.0129427, -0.0123246, -0.0129431, -0.0121838, -0.0007444, 0.0006042
8: -0.0092650, -0.0077523, -0.0092780, -0.0076710, -0.0011637, 0.0010552
9: -0.0005213, 0.0002472, -0.0005546, 0.0002542, -0.0005344, 0.0005844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003270
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003319
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010137, -0.0004429, -0.0010794, -0.0004497, -0.0004228, 0.0004861
1: -0.0042085, -0.0039923, -0.0042176, -0.0039944, -0.0001740, 0.0001652
2: 0.0131964, 0.0139968, 0.0131225, 0.0139863, -0.0005751, 0.0006403
3: 1.0084368, 1.0089853, 1.0084351, 1.0090078, -0.0004831, 0.0005383
4: -0.0038735, -0.0037361, -0.0038715, -0.0037264, -0.0001034, 0.0000967
5: 0.0031668, 0.0036096, 0.0031173, 0.0036042, -0.0003264, 0.0003735
6: -0.0024329, -0.0023855, -0.0024366, -0.0023837, -0.0000492, 0.0000510
7: -0.0129439, -0.0123941, -0.0129431, -0.0122037, -0.0007241, 0.0005343
8: -0.0092991, -0.0077684, -0.0092763, -0.0076795, -0.0011150, 0.0010717
9: -0.0005170, 0.0002655, -0.0005514, 0.0002533, -0.0005476, 0.0005517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003272
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003321
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010284, -0.0004524, -0.0010865, -0.0004491, -0.0004348, 0.0004995
1: -0.0042089, -0.0039960, -0.0042184, -0.0039941, -0.0001722, 0.0001793
2: 0.0131819, 0.0139821, 0.0131150, 0.0139872, -0.0005818, 0.0006587
3: 1.0084403, 1.0089861, 1.0084337, 1.0090098, -0.0005287, 0.0005400
4: -0.0038707, -0.0037347, -0.0038717, -0.0037255, -0.0001070, 0.0000961
5: 0.0031559, 0.0036020, 0.0031120, 0.0036046, -0.0003348, 0.0003839
6: -0.0024342, -0.0023855, -0.0024371, -0.0023835, -0.0000507, 0.0000517
7: -0.0129427, -0.0123237, -0.0129431, -0.0121834, -0.0007448, 0.0006049
8: -0.0092672, -0.0077580, -0.0092783, -0.0076713, -0.0011645, 0.0010574
9: -0.0005188, 0.0002484, -0.0005544, 0.0002544, -0.0005356, 0.0005850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003272
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003320
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0010917, -0.0004548, -0.0010771, -0.0004732, -0.0004741, 0.0004835
1: -0.0042194, -0.0039968, -0.0042220, -0.0040068, -0.0001592, 0.0001884
2: 0.0131091, 0.0139784, 0.0131204, 0.0139501, -0.0006173, 0.0006500
3: 1.0084399, 1.0090123, 1.0084703, 1.0090185, -0.0005786, 0.0004840
4: -0.0038700, -0.0037247, -0.0038648, -0.0037251, -0.0001073, 0.0000985
5: 0.0031082, 0.0036001, 0.0031187, 0.0035857, -0.0003636, 0.0003725
6: -0.0024367, -0.0023833, -0.0024322, -0.0023828, -0.0000539, 0.0000489
7: -0.0129425, -0.0121718, -0.0129403, -0.0122588, -0.0006688, 0.0007531
8: -0.0092592, -0.0076630, -0.0091979, -0.0076567, -0.0011747, 0.0010532
9: -0.0005580, 0.0002442, -0.0005670, 0.0002114, -0.0005169, 0.0005955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002002, upper bound: 0.0002036
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003172, upper bound: 0.0003039
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0010917, -0.0004548, -0.0010823, -0.0004714, -0.0004767, 0.0004889
1: -0.0042193, -0.0039968, -0.0042217, -0.0040061, -0.0001599, 0.0001898
2: 0.0131091, 0.0139785, 0.0131146, 0.0139530, -0.0006211, 0.0006559
3: 1.0084395, 1.0090122, 1.0084664, 1.0090178, -0.0005783, 0.0004881
4: -0.0038701, -0.0037247, -0.0038653, -0.0037245, -0.0001081, 0.0000992
5: 0.0031082, 0.0036002, 0.0031148, 0.0035871, -0.0003657, 0.0003766
6: -0.0024367, -0.0023833, -0.0024326, -0.0023828, -0.0000539, 0.0000493
7: -0.0129425, -0.0121716, -0.0129405, -0.0122452, -0.0006831, 0.0007536
8: -0.0092595, -0.0076634, -0.0092041, -0.0076551, -0.0011841, 0.0010613
9: -0.0005578, 0.0002443, -0.0005659, 0.0002147, -0.0005212, 0.0006005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002096, upper bound: 0.0002111
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003178, upper bound: 0.0003038
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0010917, -0.0004548, -0.0010633, -0.0004680, -0.0004827, 0.0004692
1: -0.0042194, -0.0039968, -0.0042203, -0.0040047, -0.0001627, 0.0001877
2: 0.0131091, 0.0139784, 0.0131349, 0.0139582, -0.0006304, 0.0006345
3: 1.0084399, 1.0090123, 1.0084674, 1.0090146, -0.0005747, 0.0004857
4: -0.0038700, -0.0037247, -0.0038663, -0.0037268, -0.0001058, 0.0001009
5: 0.0031082, 0.0036001, 0.0031290, 0.0035898, -0.0003704, 0.0003618
6: -0.0024367, -0.0023833, -0.0024320, -0.0023831, -0.0000536, 0.0000487
7: -0.0129425, -0.0121718, -0.0129409, -0.0122942, -0.0006335, 0.0007541
8: -0.0092592, -0.0076630, -0.0092154, -0.0076729, -0.0011641, 0.0010817
9: -0.0005580, 0.0002442, -0.0005605, 0.0002208, -0.0005321, 0.0005928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003066, upper bound: 0.0002940
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003066, upper bound: 0.0002943
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0010917, -0.0004548, -0.0010678, -0.0004667, -0.0004838, 0.0004740
1: -0.0042193, -0.0039968, -0.0042200, -0.0040040, -0.0001636, 0.0001892
2: 0.0131091, 0.0139785, 0.0131306, 0.0139602, -0.0006319, 0.0006398
3: 1.0084395, 1.0090122, 1.0084640, 1.0090137, -0.0005741, 0.0004891
4: -0.0038701, -0.0037247, -0.0038667, -0.0037266, -0.0001066, 0.0001012
5: 0.0031082, 0.0036002, 0.0031256, 0.0035908, -0.0003712, 0.0003653
6: -0.0024367, -0.0023833, -0.0024324, -0.0023832, -0.0000535, 0.0000490
7: -0.0129425, -0.0121716, -0.0129411, -0.0122834, -0.0006443, 0.0007544
8: -0.0092595, -0.0076634, -0.0092198, -0.0076729, -0.0011729, 0.0010847
9: -0.0005578, 0.0002443, -0.0005596, 0.0002231, -0.0005337, 0.0005980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0002954
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0002966
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010818, -0.0004569, -0.0011084, -0.0004814, -0.0004726, 0.0005060
1: -0.0042179, -0.0039979, -0.0042253, -0.0040095, -0.0001733, 0.0001857
2: 0.0131196, 0.0139753, 0.0130853, 0.0139376, -0.0006205, 0.0006655
3: 1.0084426, 1.0090085, 1.0084698, 1.0090269, -0.0005729, 0.0005357
4: -0.0038695, -0.0037261, -0.0038625, -0.0037207, -0.0001081, 0.0001004
5: 0.0031155, 0.0035985, 0.0030951, 0.0035793, -0.0003630, 0.0003886
6: -0.0024362, -0.0023836, -0.0024350, -0.0023821, -0.0000541, 0.0000514
7: -0.0129422, -0.0121967, -0.0129394, -0.0121501, -0.0007778, 0.0007288
8: -0.0092525, -0.0076765, -0.0091709, -0.0076198, -0.0011740, 0.0010880
9: -0.0005524, 0.0002406, -0.0005795, 0.0001970, -0.0005432, 0.0005873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002074, upper bound: 0.0002233
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003189, upper bound: 0.0003121
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010826, -0.0004561, -0.0011085, -0.0004813, -0.0004757, 0.0005079
1: -0.0042174, -0.0039968, -0.0042252, -0.0040095, -0.0001746, 0.0001866
2: 0.0131203, 0.0139765, 0.0130853, 0.0139377, -0.0006245, 0.0006684
3: 1.0084393, 1.0090072, 1.0084696, 1.0090268, -0.0005758, 0.0005376
4: -0.0038697, -0.0037264, -0.0038625, -0.0037207, -0.0001086, 0.0001011
5: 0.0031151, 0.0035992, 0.0030951, 0.0035793, -0.0003653, 0.0003901
6: -0.0024365, -0.0023837, -0.0024350, -0.0023821, -0.0000544, 0.0000513
7: -0.0129423, -0.0121910, -0.0129394, -0.0121500, -0.0007781, 0.0007347
8: -0.0092551, -0.0076804, -0.0091710, -0.0076198, -0.0011803, 0.0010969
9: -0.0005504, 0.0002420, -0.0005794, 0.0001971, -0.0005480, 0.0005905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002124, upper bound: 0.0002241
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003190, upper bound: 0.0003130
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0010983, -0.0004542, -0.0010816, -0.0004789, -0.0004952, 0.0004862
1: -0.0042202, -0.0039965, -0.0042211, -0.0040085, -0.0001765, 0.0001866
2: 0.0131019, 0.0139793, 0.0131165, 0.0139413, -0.0006480, 0.0006451
3: 1.0084385, 1.0090144, 1.0084692, 1.0090163, -0.0005778, 0.0005308
4: -0.0038702, -0.0037237, -0.0038631, -0.0037249, -0.0001060, 0.0001041
5: 0.0031032, 0.0036006, 0.0031154, 0.0035812, -0.0003802, 0.0003740
6: -0.0024373, -0.0023831, -0.0024340, -0.0023830, -0.0000543, 0.0000509
7: -0.0129425, -0.0121518, -0.0129396, -0.0122152, -0.0007135, 0.0007744
8: -0.0092612, -0.0076546, -0.0091789, -0.0076594, -0.0011554, 0.0011282
9: -0.0005610, 0.0002452, -0.0005638, 0.0002013, -0.0005642, 0.0005825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003088
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003095
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0010983, -0.0004542, -0.0010814, -0.0004769, -0.0004959, 0.0004860
1: -0.0042201, -0.0039964, -0.0042208, -0.0040078, -0.0001765, 0.0001873
2: 0.0131020, 0.0139794, 0.0131161, 0.0139445, -0.0006487, 0.0006455
3: 1.0084385, 1.0090141, 1.0084671, 1.0090158, -0.0005773, 0.0005320
4: -0.0038702, -0.0037238, -0.0038637, -0.0037248, -0.0001062, 0.0001042
5: 0.0031032, 0.0036007, 0.0031155, 0.0035828, -0.0003806, 0.0003738
6: -0.0024373, -0.0023831, -0.0024340, -0.0023830, -0.0000543, 0.0000509
7: -0.0129425, -0.0121516, -0.0129399, -0.0122171, -0.0007119, 0.0007746
8: -0.0092614, -0.0076550, -0.0091858, -0.0076589, -0.0011574, 0.0011290
9: -0.0005609, 0.0002454, -0.0005629, 0.0002050, -0.0005649, 0.0005849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003088
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003095
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010111, -0.0004433, -0.0011442, -0.0004807, -0.0003948, 0.0005547
1: -0.0042088, -0.0039928, -0.0042310, -0.0040090, -0.0001669, 0.0001842
2: 0.0131988, 0.0139961, 0.0130447, 0.0139387, -0.0005351, 0.0007266
3: 1.0084395, 1.0089859, 1.0084655, 1.0090411, -0.0005280, 0.0005203
4: -0.0038733, -0.0037362, -0.0038626, -0.0037152, -0.0001171, 0.0000896
5: 0.0031687, 0.0036092, 0.0030683, 0.0035798, -0.0003045, 0.0004258
6: -0.0024326, -0.0023855, -0.0024369, -0.0023809, -0.0000517, 0.0000514
7: -0.0129438, -0.0124041, -0.0129394, -0.0120691, -0.0008592, 0.0005213
8: -0.0092976, -0.0077673, -0.0091731, -0.0075683, -0.0012608, 0.0009927
9: -0.0005180, 0.0002647, -0.0006008, 0.0001982, -0.0005067, 0.0006213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002992, upper bound: 0.0003026
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002992, upper bound: 0.0003026
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010111, -0.0004433, -0.0011324, -0.0004756, -0.0004012, 0.0005376
1: -0.0042088, -0.0039928, -0.0042291, -0.0040065, -0.0001672, 0.0001806
2: 0.0131988, 0.0139961, 0.0130576, 0.0139464, -0.0005430, 0.0007052
3: 1.0084395, 1.0089859, 1.0084609, 1.0090365, -0.0005201, 0.0005250
4: -0.0038733, -0.0037362, -0.0038641, -0.0037169, -0.0001139, 0.0000909
5: 0.0031687, 0.0036092, 0.0030770, 0.0035838, -0.0003095, 0.0004127
6: -0.0024326, -0.0023855, -0.0024367, -0.0023813, -0.0000513, 0.0000512
7: -0.0129438, -0.0124041, -0.0129400, -0.0120990, -0.0008288, 0.0005221
8: -0.0092976, -0.0077673, -0.0091899, -0.0075848, -0.0012279, 0.0010049
9: -0.0005180, 0.0002647, -0.0005940, 0.0002071, -0.0005120, 0.0006053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002992, upper bound: 0.0003026
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002992, upper bound: 0.0003026
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010137, -0.0004429, -0.0011445, -0.0004806, -0.0003988, 0.0005572
1: -0.0042085, -0.0039923, -0.0042309, -0.0040089, -0.0001682, 0.0001848
2: 0.0131964, 0.0139968, 0.0130446, 0.0139388, -0.0005400, 0.0007303
3: 1.0084368, 1.0089853, 1.0084654, 1.0090411, -0.0005314, 0.0005199
4: -0.0038735, -0.0037361, -0.0038627, -0.0037152, -0.0001178, 0.0000904
5: 0.0031668, 0.0036096, 0.0030681, 0.0035798, -0.0003076, 0.0004278
6: -0.0024329, -0.0023855, -0.0024369, -0.0023809, -0.0000520, 0.0000513
7: -0.0129439, -0.0123941, -0.0129394, -0.0120686, -0.0008599, 0.0005313
8: -0.0092991, -0.0077684, -0.0091733, -0.0075682, -0.0012691, 0.0010015
9: -0.0005170, 0.0002655, -0.0006008, 0.0001983, -0.0005112, 0.0006256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003046
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003046
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010137, -0.0004429, -0.0011325, -0.0004756, -0.0004047, 0.0005409
1: -0.0042085, -0.0039923, -0.0042291, -0.0040065, -0.0001682, 0.0001814
2: 0.0131964, 0.0139968, 0.0130574, 0.0139465, -0.0005473, 0.0007103
3: 1.0084368, 1.0089853, 1.0084609, 1.0090367, -0.0005235, 0.0005244
4: -0.0038735, -0.0037361, -0.0038641, -0.0037169, -0.0001149, 0.0000916
5: 0.0031668, 0.0036096, 0.0030770, 0.0035838, -0.0003121, 0.0004154
6: -0.0024329, -0.0023855, -0.0024367, -0.0023813, -0.0000516, 0.0000512
7: -0.0129439, -0.0123941, -0.0129400, -0.0120985, -0.0008297, 0.0005321
8: -0.0092991, -0.0077684, -0.0091900, -0.0075846, -0.0012392, 0.0010115
9: -0.0005170, 0.0002655, -0.0005940, 0.0002072, -0.0005155, 0.0006112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003046
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003046
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010300, -0.0004531, -0.0011036, -0.0004697, -0.0004295, 0.0005165
1: -0.0042096, -0.0039965, -0.0042257, -0.0040049, -0.0001583, 0.0001900
2: 0.0131791, 0.0139810, 0.0130903, 0.0139556, -0.0005718, 0.0006894
3: 1.0084426, 1.0089878, 1.0084643, 1.0090281, -0.0005527, 0.0004926
4: -0.0038705, -0.0037341, -0.0038658, -0.0037212, -0.0001132, 0.0000933
5: 0.0031546, 0.0036015, 0.0030987, 0.0035884, -0.0003305, 0.0003975
6: -0.0024341, -0.0023853, -0.0024337, -0.0023820, -0.0000521, 0.0000484
7: -0.0129427, -0.0123246, -0.0129407, -0.0121915, -0.0007374, 0.0006037
8: -0.0092650, -0.0077523, -0.0092097, -0.0076218, -0.0012373, 0.0010110
9: -0.0005213, 0.0002472, -0.0005809, 0.0002177, -0.0005044, 0.0006245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002915
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002915
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010300, -0.0004531, -0.0011170, -0.0004803, -0.0004135, 0.0005265
1: -0.0042096, -0.0039965, -0.0042266, -0.0040090, -0.0001633, 0.0001880
2: 0.0131791, 0.0139810, 0.0130755, 0.0139392, -0.0005488, 0.0006935
3: 1.0084426, 1.0089878, 1.0084674, 1.0090302, -0.0005529, 0.0005196
4: -0.0038705, -0.0037341, -0.0038627, -0.0037193, -0.0001129, 0.0000899
5: 0.0031546, 0.0036015, 0.0030886, 0.0035801, -0.0003181, 0.0004044
6: -0.0024341, -0.0023853, -0.0024357, -0.0023818, -0.0000523, 0.0000504
7: -0.0129427, -0.0123246, -0.0129395, -0.0121267, -0.0008030, 0.0006017
8: -0.0092650, -0.0077523, -0.0091743, -0.0076076, -0.0012265, 0.0009851
9: -0.0005213, 0.0002472, -0.0005844, 0.0001988, -0.0004970, 0.0006135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002992
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002992
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010284, -0.0004524, -0.0011039, -0.0004696, -0.0004299, 0.0005172
1: -0.0042089, -0.0039960, -0.0042257, -0.0040049, -0.0001586, 0.0001900
2: 0.0131819, 0.0139821, 0.0130900, 0.0139557, -0.0005718, 0.0006904
3: 1.0084403, 1.0089861, 1.0084642, 1.0090280, -0.0005542, 0.0004935
4: -0.0038707, -0.0037347, -0.0038658, -0.0037212, -0.0001134, 0.0000934
5: 0.0031559, 0.0036020, 0.0030985, 0.0035885, -0.0003308, 0.0003981
6: -0.0024342, -0.0023855, -0.0024337, -0.0023820, -0.0000522, 0.0000483
7: -0.0129427, -0.0123237, -0.0129407, -0.0121907, -0.0007382, 0.0006043
8: -0.0092672, -0.0077580, -0.0092100, -0.0076220, -0.0012392, 0.0010120
9: -0.0005188, 0.0002484, -0.0005808, 0.0002179, -0.0005053, 0.0006252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002930
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002930
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010284, -0.0004524, -0.0011170, -0.0004803, -0.0004138, 0.0005276
1: -0.0042089, -0.0039960, -0.0042265, -0.0040089, -0.0001637, 0.0001883
2: 0.0131819, 0.0139821, 0.0130755, 0.0139393, -0.0005495, 0.0006955
3: 1.0084403, 1.0089861, 1.0084674, 1.0090302, -0.0005547, 0.0005187
4: -0.0038707, -0.0037347, -0.0038628, -0.0037193, -0.0001133, 0.0000901
5: 0.0031559, 0.0036020, 0.0030886, 0.0035801, -0.0003183, 0.0004053
6: -0.0024342, -0.0023855, -0.0024357, -0.0023818, -0.0000523, 0.0000503
7: -0.0129427, -0.0123237, -0.0129395, -0.0121266, -0.0008033, 0.0006025
8: -0.0092672, -0.0077580, -0.0091745, -0.0076076, -0.0012306, 0.0009875
9: -0.0005188, 0.0002484, -0.0005843, 0.0001989, -0.0004982, 0.0006156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002996
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002996
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010771, -0.0004732, -0.0010917, -0.0004548, -0.0004835, 0.0004741
1: -0.0042220, -0.0040068, -0.0042194, -0.0039968, -0.0001884, 0.0001592
2: 0.0131204, 0.0139501, 0.0131091, 0.0139784, -0.0006500, 0.0006173
3: 1.0084703, 1.0090185, 1.0084399, 1.0090123, -0.0004840, 0.0005786
4: -0.0038648, -0.0037251, -0.0038700, -0.0037247, -0.0000985, 0.0001073
5: 0.0031187, 0.0035857, 0.0031082, 0.0036001, -0.0003725, 0.0003636
6: -0.0024322, -0.0023828, -0.0024367, -0.0023833, -0.0000489, 0.0000539
7: -0.0129403, -0.0122588, -0.0129425, -0.0121718, -0.0007531, 0.0006688
8: -0.0091979, -0.0076567, -0.0092592, -0.0076630, -0.0010532, 0.0011747
9: -0.0005670, 0.0002114, -0.0005580, 0.0002442, -0.0005955, 0.0005169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002036, upper bound: 0.0002002
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003039, upper bound: 0.0003172
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010823, -0.0004714, -0.0010917, -0.0004548, -0.0004889, 0.0004767
1: -0.0042217, -0.0040061, -0.0042193, -0.0039968, -0.0001898, 0.0001599
2: 0.0131146, 0.0139530, 0.0131091, 0.0139785, -0.0006559, 0.0006211
3: 1.0084664, 1.0090178, 1.0084395, 1.0090122, -0.0004881, 0.0005783
4: -0.0038653, -0.0037245, -0.0038701, -0.0037247, -0.0000992, 0.0001081
5: 0.0031148, 0.0035871, 0.0031082, 0.0036002, -0.0003766, 0.0003657
6: -0.0024326, -0.0023828, -0.0024367, -0.0023833, -0.0000493, 0.0000539
7: -0.0129405, -0.0122452, -0.0129425, -0.0121716, -0.0007536, 0.0006831
8: -0.0092041, -0.0076551, -0.0092595, -0.0076634, -0.0010613, 0.0011841
9: -0.0005659, 0.0002147, -0.0005578, 0.0002443, -0.0006005, 0.0005212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002111, upper bound: 0.0002096
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003178
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010633, -0.0004680, -0.0010917, -0.0004548, -0.0004692, 0.0004827
1: -0.0042203, -0.0040047, -0.0042194, -0.0039968, -0.0001877, 0.0001627
2: 0.0131349, 0.0139582, 0.0131091, 0.0139784, -0.0006345, 0.0006304
3: 1.0084674, 1.0090146, 1.0084399, 1.0090123, -0.0004857, 0.0005747
4: -0.0038663, -0.0037268, -0.0038700, -0.0037247, -0.0001009, 0.0001058
5: 0.0031290, 0.0035898, 0.0031082, 0.0036001, -0.0003618, 0.0003704
6: -0.0024320, -0.0023831, -0.0024367, -0.0023833, -0.0000487, 0.0000536
7: -0.0129409, -0.0122942, -0.0129425, -0.0121718, -0.0007541, 0.0006335
8: -0.0092154, -0.0076729, -0.0092592, -0.0076630, -0.0010817, 0.0011641
9: -0.0005605, 0.0002208, -0.0005580, 0.0002442, -0.0005928, 0.0005321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002940, upper bound: 0.0003066
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002940, upper bound: 0.0003066
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010678, -0.0004667, -0.0010917, -0.0004548, -0.0004740, 0.0004838
1: -0.0042200, -0.0040040, -0.0042193, -0.0039968, -0.0001892, 0.0001636
2: 0.0131306, 0.0139602, 0.0131091, 0.0139785, -0.0006398, 0.0006319
3: 1.0084640, 1.0090137, 1.0084395, 1.0090122, -0.0004891, 0.0005741
4: -0.0038667, -0.0037266, -0.0038701, -0.0037247, -0.0001012, 0.0001066
5: 0.0031256, 0.0035908, 0.0031082, 0.0036002, -0.0003653, 0.0003712
6: -0.0024324, -0.0023832, -0.0024367, -0.0023833, -0.0000490, 0.0000535
7: -0.0129411, -0.0122834, -0.0129425, -0.0121716, -0.0007544, 0.0006443
8: -0.0092198, -0.0076729, -0.0092595, -0.0076634, -0.0010847, 0.0011729
9: -0.0005596, 0.0002231, -0.0005578, 0.0002443, -0.0005980, 0.0005337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0003088
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0003098
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011084, -0.0004814, -0.0010818, -0.0004569, -0.0005060, 0.0004726
1: -0.0042253, -0.0040095, -0.0042179, -0.0039979, -0.0001857, 0.0001733
2: 0.0130853, 0.0139376, 0.0131196, 0.0139753, -0.0006655, 0.0006205
3: 1.0084698, 1.0090269, 1.0084426, 1.0090085, -0.0005357, 0.0005729
4: -0.0038625, -0.0037207, -0.0038695, -0.0037261, -0.0001004, 0.0001081
5: 0.0030951, 0.0035793, 0.0031155, 0.0035985, -0.0003886, 0.0003630
6: -0.0024350, -0.0023821, -0.0024362, -0.0023836, -0.0000514, 0.0000541
7: -0.0129394, -0.0121501, -0.0129422, -0.0121967, -0.0007288, 0.0007778
8: -0.0091709, -0.0076198, -0.0092525, -0.0076765, -0.0010880, 0.0011740
9: -0.0005795, 0.0001970, -0.0005524, 0.0002406, -0.0005873, 0.0005432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002233, upper bound: 0.0002074
time: 0.49 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003121, upper bound: 0.0003189
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011085, -0.0004813, -0.0010826, -0.0004561, -0.0005079, 0.0004757
1: -0.0042252, -0.0040095, -0.0042174, -0.0039968, -0.0001866, 0.0001746
2: 0.0130853, 0.0139377, 0.0131203, 0.0139765, -0.0006684, 0.0006245
3: 1.0084696, 1.0090268, 1.0084393, 1.0090072, -0.0005376, 0.0005758
4: -0.0038625, -0.0037207, -0.0038697, -0.0037264, -0.0001011, 0.0001086
5: 0.0030951, 0.0035793, 0.0031151, 0.0035992, -0.0003901, 0.0003653
6: -0.0024350, -0.0023821, -0.0024365, -0.0023837, -0.0000513, 0.0000544
7: -0.0129394, -0.0121500, -0.0129423, -0.0121910, -0.0007347, 0.0007781
8: -0.0091710, -0.0076198, -0.0092551, -0.0076804, -0.0010969, 0.0011803
9: -0.0005794, 0.0001971, -0.0005504, 0.0002420, -0.0005905, 0.0005480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002241, upper bound: 0.0002124
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003130, upper bound: 0.0003190
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010816, -0.0004789, -0.0010983, -0.0004542, -0.0004862, 0.0004952
1: -0.0042211, -0.0040085, -0.0042202, -0.0039965, -0.0001866, 0.0001765
2: 0.0131165, 0.0139413, 0.0131019, 0.0139793, -0.0006451, 0.0006480
3: 1.0084692, 1.0090163, 1.0084385, 1.0090144, -0.0005308, 0.0005778
4: -0.0038631, -0.0037249, -0.0038702, -0.0037237, -0.0001041, 0.0001060
5: 0.0031154, 0.0035812, 0.0031032, 0.0036006, -0.0003740, 0.0003802
6: -0.0024340, -0.0023830, -0.0024373, -0.0023831, -0.0000509, 0.0000543
7: -0.0129396, -0.0122152, -0.0129425, -0.0121518, -0.0007744, 0.0007135
8: -0.0091789, -0.0076594, -0.0092612, -0.0076546, -0.0011282, 0.0011554
9: -0.0005638, 0.0002013, -0.0005610, 0.0002452, -0.0005825, 0.0005642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010814, -0.0004769, -0.0010983, -0.0004542, -0.0004860, 0.0004959
1: -0.0042208, -0.0040078, -0.0042201, -0.0039964, -0.0001873, 0.0001765
2: 0.0131161, 0.0139445, 0.0131020, 0.0139794, -0.0006455, 0.0006487
3: 1.0084671, 1.0090158, 1.0084385, 1.0090141, -0.0005320, 0.0005773
4: -0.0038637, -0.0037248, -0.0038702, -0.0037238, -0.0001042, 0.0001062
5: 0.0031155, 0.0035828, 0.0031032, 0.0036007, -0.0003738, 0.0003806
6: -0.0024340, -0.0023830, -0.0024373, -0.0023831, -0.0000509, 0.0000543
7: -0.0129399, -0.0122171, -0.0129425, -0.0121516, -0.0007746, 0.0007119
8: -0.0091858, -0.0076589, -0.0092614, -0.0076550, -0.0011290, 0.0011574
9: -0.0005629, 0.0002050, -0.0005609, 0.0002454, -0.0005849, 0.0005649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011442, -0.0004807, -0.0010111, -0.0004433, -0.0005547, 0.0003948
1: -0.0042310, -0.0040090, -0.0042088, -0.0039928, -0.0001842, 0.0001669
2: 0.0130447, 0.0139387, 0.0131988, 0.0139961, -0.0007266, 0.0005351
3: 1.0084655, 1.0090411, 1.0084395, 1.0089859, -0.0005203, 0.0005280
4: -0.0038626, -0.0037152, -0.0038733, -0.0037362, -0.0000896, 0.0001171
5: 0.0030683, 0.0035798, 0.0031687, 0.0036092, -0.0004258, 0.0003045
6: -0.0024369, -0.0023809, -0.0024326, -0.0023855, -0.0000514, 0.0000517
7: -0.0129394, -0.0120691, -0.0129438, -0.0124041, -0.0005213, 0.0008592
8: -0.0091731, -0.0075683, -0.0092976, -0.0077673, -0.0009927, 0.0012608
9: -0.0006008, 0.0001982, -0.0005180, 0.0002647, -0.0006213, 0.0005067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0002992
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011324, -0.0004756, -0.0010111, -0.0004433, -0.0005376, 0.0004012
1: -0.0042291, -0.0040065, -0.0042088, -0.0039928, -0.0001806, 0.0001672
2: 0.0130576, 0.0139464, 0.0131988, 0.0139961, -0.0007052, 0.0005430
3: 1.0084609, 1.0090365, 1.0084395, 1.0089859, -0.0005250, 0.0005201
4: -0.0038641, -0.0037169, -0.0038733, -0.0037362, -0.0000909, 0.0001139
5: 0.0030770, 0.0035838, 0.0031687, 0.0036092, -0.0004127, 0.0003095
6: -0.0024367, -0.0023813, -0.0024326, -0.0023855, -0.0000512, 0.0000513
7: -0.0129400, -0.0120990, -0.0129438, -0.0124041, -0.0005221, 0.0008288
8: -0.0091899, -0.0075848, -0.0092976, -0.0077673, -0.0010049, 0.0012279
9: -0.0005940, 0.0002071, -0.0005180, 0.0002647, -0.0006053, 0.0005120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0002992
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011445, -0.0004806, -0.0010137, -0.0004429, -0.0005572, 0.0003988
1: -0.0042309, -0.0040089, -0.0042085, -0.0039923, -0.0001848, 0.0001682
2: 0.0130446, 0.0139388, 0.0131964, 0.0139968, -0.0007303, 0.0005400
3: 1.0084654, 1.0090411, 1.0084368, 1.0089853, -0.0005199, 0.0005314
4: -0.0038627, -0.0037152, -0.0038735, -0.0037361, -0.0000904, 0.0001178
5: 0.0030681, 0.0035798, 0.0031668, 0.0036096, -0.0004278, 0.0003076
6: -0.0024369, -0.0023809, -0.0024329, -0.0023855, -0.0000513, 0.0000520
7: -0.0129394, -0.0120686, -0.0129439, -0.0123941, -0.0005313, 0.0008599
8: -0.0091733, -0.0075682, -0.0092991, -0.0077684, -0.0010015, 0.0012691
9: -0.0006008, 0.0001983, -0.0005170, 0.0002655, -0.0006256, 0.0005112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003035
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011325, -0.0004756, -0.0010137, -0.0004429, -0.0005409, 0.0004047
1: -0.0042291, -0.0040065, -0.0042085, -0.0039923, -0.0001814, 0.0001682
2: 0.0130574, 0.0139465, 0.0131964, 0.0139968, -0.0007103, 0.0005473
3: 1.0084609, 1.0090367, 1.0084368, 1.0089853, -0.0005244, 0.0005235
4: -0.0038641, -0.0037169, -0.0038735, -0.0037361, -0.0000916, 0.0001149
5: 0.0030770, 0.0035838, 0.0031668, 0.0036096, -0.0004154, 0.0003121
6: -0.0024367, -0.0023813, -0.0024329, -0.0023855, -0.0000512, 0.0000516
7: -0.0129400, -0.0120985, -0.0129439, -0.0123941, -0.0005321, 0.0008297
8: -0.0091900, -0.0075846, -0.0092991, -0.0077684, -0.0010115, 0.0012392
9: -0.0005940, 0.0002072, -0.0005170, 0.0002655, -0.0006112, 0.0005155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003035
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011036, -0.0004697, -0.0010300, -0.0004531, -0.0005165, 0.0004295
1: -0.0042257, -0.0040049, -0.0042096, -0.0039965, -0.0001900, 0.0001583
2: 0.0130903, 0.0139556, 0.0131791, 0.0139810, -0.0006894, 0.0005718
3: 1.0084643, 1.0090281, 1.0084426, 1.0089878, -0.0004926, 0.0005527
4: -0.0038658, -0.0037212, -0.0038705, -0.0037341, -0.0000933, 0.0001132
5: 0.0030987, 0.0035884, 0.0031546, 0.0036015, -0.0003975, 0.0003305
6: -0.0024337, -0.0023820, -0.0024341, -0.0023853, -0.0000484, 0.0000521
7: -0.0129407, -0.0121915, -0.0129427, -0.0123246, -0.0006037, 0.0007374
8: -0.0092097, -0.0076218, -0.0092650, -0.0077523, -0.0010110, 0.0012373
9: -0.0005809, 0.0002177, -0.0005213, 0.0002472, -0.0006245, 0.0005044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003085
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003085
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011170, -0.0004803, -0.0010300, -0.0004531, -0.0005265, 0.0004135
1: -0.0042266, -0.0040090, -0.0042096, -0.0039965, -0.0001880, 0.0001633
2: 0.0130755, 0.0139392, 0.0131791, 0.0139810, -0.0006935, 0.0005488
3: 1.0084674, 1.0090302, 1.0084426, 1.0089878, -0.0005196, 0.0005529
4: -0.0038627, -0.0037193, -0.0038705, -0.0037341, -0.0000899, 0.0001129
5: 0.0030886, 0.0035801, 0.0031546, 0.0036015, -0.0004044, 0.0003181
6: -0.0024357, -0.0023818, -0.0024341, -0.0023853, -0.0000504, 0.0000523
7: -0.0129395, -0.0121267, -0.0129427, -0.0123246, -0.0006017, 0.0008030
8: -0.0091743, -0.0076076, -0.0092650, -0.0077523, -0.0009851, 0.0012265
9: -0.0005844, 0.0001988, -0.0005213, 0.0002472, -0.0006135, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003079
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003079
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011039, -0.0004696, -0.0010284, -0.0004524, -0.0005172, 0.0004299
1: -0.0042257, -0.0040049, -0.0042089, -0.0039960, -0.0001900, 0.0001586
2: 0.0130900, 0.0139557, 0.0131819, 0.0139821, -0.0006904, 0.0005718
3: 1.0084642, 1.0090280, 1.0084403, 1.0089861, -0.0004935, 0.0005542
4: -0.0038658, -0.0037212, -0.0038707, -0.0037347, -0.0000934, 0.0001134
5: 0.0030985, 0.0035885, 0.0031559, 0.0036020, -0.0003981, 0.0003308
6: -0.0024337, -0.0023820, -0.0024342, -0.0023855, -0.0000483, 0.0000522
7: -0.0129407, -0.0121907, -0.0129427, -0.0123237, -0.0006043, 0.0007382
8: -0.0092100, -0.0076220, -0.0092672, -0.0077580, -0.0010120, 0.0012392
9: -0.0005808, 0.0002179, -0.0005188, 0.0002484, -0.0006252, 0.0005053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003096
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003096
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011170, -0.0004803, -0.0010284, -0.0004524, -0.0005276, 0.0004138
1: -0.0042265, -0.0040089, -0.0042089, -0.0039960, -0.0001883, 0.0001637
2: 0.0130755, 0.0139393, 0.0131819, 0.0139821, -0.0006955, 0.0005495
3: 1.0084674, 1.0090302, 1.0084403, 1.0089861, -0.0005187, 0.0005547
4: -0.0038628, -0.0037193, -0.0038707, -0.0037347, -0.0000901, 0.0001133
5: 0.0030886, 0.0035801, 0.0031559, 0.0036020, -0.0004053, 0.0003183
6: -0.0024357, -0.0023818, -0.0024342, -0.0023855, -0.0000503, 0.0000523
7: -0.0129395, -0.0121266, -0.0129427, -0.0123237, -0.0006025, 0.0008033
8: -0.0091745, -0.0076076, -0.0092672, -0.0077580, -0.0009875, 0.0012306
9: -0.0005843, 0.0001989, -0.0005188, 0.0002484, -0.0006156, 0.0004982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003080
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003080
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010771, -0.0004732, -0.0011442, -0.0004807, -0.0004522, 0.0005128
1: -0.0042220, -0.0040068, -0.0042310, -0.0040090, -0.0001763, 0.0001681
2: 0.0131204, 0.0139501, 0.0130447, 0.0139387, -0.0005973, 0.0006600
3: 1.0084703, 1.0090185, 1.0084655, 1.0090411, -0.0005163, 0.0005530
4: -0.0038648, -0.0037251, -0.0038626, -0.0037152, -0.0001041, 0.0000972
5: 0.0031187, 0.0035857, 0.0030683, 0.0035798, -0.0003476, 0.0003927
6: -0.0024322, -0.0023828, -0.0024369, -0.0023809, -0.0000513, 0.0000541
7: -0.0129403, -0.0122588, -0.0129394, -0.0120691, -0.0008545, 0.0006654
8: -0.0091979, -0.0076567, -0.0091731, -0.0075683, -0.0011057, 0.0010603
9: -0.0005670, 0.0002114, -0.0006008, 0.0001982, -0.0005338, 0.0005384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002758, upper bound: 0.0002921
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002758, upper bound: 0.0002921
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010771, -0.0004732, -0.0011324, -0.0004756, -0.0004641, 0.0005034
1: -0.0042220, -0.0040068, -0.0042291, -0.0040065, -0.0001805, 0.0001685
2: 0.0131204, 0.0139501, 0.0130576, 0.0139464, -0.0006157, 0.0006516
3: 1.0084703, 1.0090185, 1.0084609, 1.0090365, -0.0005172, 0.0005577
4: -0.0038648, -0.0037251, -0.0038641, -0.0037169, -0.0001031, 0.0001006
5: 0.0031187, 0.0035857, 0.0030770, 0.0035838, -0.0003570, 0.0003858
6: -0.0024322, -0.0023828, -0.0024367, -0.0023813, -0.0000509, 0.0000539
7: -0.0129403, -0.0122588, -0.0129400, -0.0120990, -0.0008248, 0.0006668
8: -0.0091979, -0.0076567, -0.0091899, -0.0075848, -0.0011008, 0.0011001
9: -0.0005670, 0.0002114, -0.0005940, 0.0002071, -0.0005551, 0.0005394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002758, upper bound: 0.0002921
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002758, upper bound: 0.0002921
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010823, -0.0004714, -0.0011445, -0.0004806, -0.0004555, 0.0005160
1: -0.0042217, -0.0040061, -0.0042309, -0.0040089, -0.0001768, 0.0001691
2: 0.0131146, 0.0139530, 0.0130446, 0.0139388, -0.0006010, 0.0006648
3: 1.0084664, 1.0090178, 1.0084654, 1.0090411, -0.0005202, 0.0005524
4: -0.0038653, -0.0037245, -0.0038627, -0.0037152, -0.0001050, 0.0000976
5: 0.0031148, 0.0035871, 0.0030681, 0.0035798, -0.0003501, 0.0003952
6: -0.0024326, -0.0023828, -0.0024369, -0.0023809, -0.0000517, 0.0000540
7: -0.0129405, -0.0122452, -0.0129394, -0.0120686, -0.0008554, 0.0006794
8: -0.0092041, -0.0076551, -0.0091733, -0.0075682, -0.0011160, 0.0010641
9: -0.0005659, 0.0002147, -0.0006008, 0.0001983, -0.0005355, 0.0005438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002944
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002945
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010823, -0.0004714, -0.0011325, -0.0004756, -0.0004674, 0.0005064
1: -0.0042217, -0.0040061, -0.0042291, -0.0040065, -0.0001810, 0.0001695
2: 0.0131146, 0.0139530, 0.0130574, 0.0139465, -0.0006193, 0.0006564
3: 1.0084664, 1.0090178, 1.0084609, 1.0090367, -0.0005211, 0.0005569
4: -0.0038653, -0.0037245, -0.0038641, -0.0037169, -0.0001040, 0.0001010
5: 0.0031148, 0.0035871, 0.0030770, 0.0035838, -0.0003595, 0.0003882
6: -0.0024326, -0.0023828, -0.0024367, -0.0023813, -0.0000513, 0.0000539
7: -0.0129405, -0.0122452, -0.0129400, -0.0120985, -0.0008257, 0.0006808
8: -0.0092041, -0.0076551, -0.0091900, -0.0075846, -0.0011111, 0.0011038
9: -0.0005659, 0.0002147, -0.0005940, 0.0002072, -0.0005567, 0.0005450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002944
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002945
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010633, -0.0004680, -0.0011442, -0.0004807, -0.0004381, 0.0005254
1: -0.0042203, -0.0040047, -0.0042310, -0.0040090, -0.0001764, 0.0001718
2: 0.0131349, 0.0139582, 0.0130447, 0.0139387, -0.0005829, 0.0006794
3: 1.0084674, 1.0090146, 1.0084655, 1.0090411, -0.0005176, 0.0005491
4: -0.0038663, -0.0037268, -0.0038626, -0.0037152, -0.0001077, 0.0000960
5: 0.0031290, 0.0035898, 0.0030683, 0.0035798, -0.0003370, 0.0004026
6: -0.0024320, -0.0023831, -0.0024369, -0.0023809, -0.0000511, 0.0000537
7: -0.0129409, -0.0122942, -0.0129394, -0.0120691, -0.0008560, 0.0006300
8: -0.0092154, -0.0076729, -0.0091731, -0.0075683, -0.0011476, 0.0010543
9: -0.0005605, 0.0002208, -0.0006008, 0.0001982, -0.0005340, 0.0005608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002844
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002844
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010633, -0.0004680, -0.0011324, -0.0004756, -0.0004458, 0.0005091
1: -0.0042203, -0.0040047, -0.0042291, -0.0040065, -0.0001769, 0.0001682
2: 0.0131349, 0.0139582, 0.0130576, 0.0139464, -0.0005943, 0.0006592
3: 1.0084674, 1.0090146, 1.0084609, 1.0090365, -0.0005100, 0.0005537
4: -0.0038663, -0.0037268, -0.0038641, -0.0037169, -0.0001046, 0.0000976
5: 0.0031290, 0.0035898, 0.0030770, 0.0035838, -0.0003431, 0.0003903
6: -0.0024320, -0.0023831, -0.0024367, -0.0023813, -0.0000507, 0.0000536
7: -0.0129409, -0.0122942, -0.0129400, -0.0120990, -0.0008256, 0.0006310
8: -0.0092154, -0.0076729, -0.0091899, -0.0075848, -0.0011154, 0.0010704
9: -0.0005605, 0.0002208, -0.0005940, 0.0002071, -0.0005411, 0.0005452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002844
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002844
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010678, -0.0004667, -0.0011445, -0.0004806, -0.0004409, 0.0005281
1: -0.0042200, -0.0040040, -0.0042309, -0.0040089, -0.0001770, 0.0001725
2: 0.0131306, 0.0139602, 0.0130446, 0.0139388, -0.0005855, 0.0006833
3: 1.0084640, 1.0090137, 1.0084654, 1.0090411, -0.0005209, 0.0005482
4: -0.0038667, -0.0037266, -0.0038627, -0.0037152, -0.0001084, 0.0000964
5: 0.0031256, 0.0035908, 0.0030681, 0.0035798, -0.0003390, 0.0004047
6: -0.0024324, -0.0023832, -0.0024369, -0.0023809, -0.0000514, 0.0000537
7: -0.0129411, -0.0122834, -0.0129394, -0.0120686, -0.0008568, 0.0006407
8: -0.0092198, -0.0076729, -0.0091733, -0.0075682, -0.0011561, 0.0010584
9: -0.0005596, 0.0002231, -0.0006008, 0.0001983, -0.0005359, 0.0005653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002876
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002878
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010678, -0.0004667, -0.0011325, -0.0004756, -0.0004480, 0.0005125
1: -0.0042200, -0.0040040, -0.0042291, -0.0040065, -0.0001772, 0.0001691
2: 0.0131306, 0.0139602, 0.0130574, 0.0139465, -0.0005967, 0.0006642
3: 1.0084640, 1.0090137, 1.0084609, 1.0090367, -0.0005133, 0.0005528
4: -0.0038667, -0.0037266, -0.0038641, -0.0037169, -0.0001055, 0.0000979
5: 0.0031256, 0.0035908, 0.0030770, 0.0035838, -0.0003448, 0.0003929
6: -0.0024324, -0.0023832, -0.0024367, -0.0023813, -0.0000510, 0.0000535
7: -0.0129411, -0.0122834, -0.0129400, -0.0120985, -0.0008265, 0.0006416
8: -0.0092198, -0.0076729, -0.0091900, -0.0075846, -0.0011262, 0.0010724
9: -0.0005596, 0.0002231, -0.0005940, 0.0002072, -0.0005420, 0.0005510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002876
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002878
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011097, -0.0004812, -0.0010975, -0.0004704, -0.0004927, 0.0004715
1: -0.0042254, -0.0040094, -0.0042245, -0.0040053, -0.0001695, 0.0001768
2: 0.0130838, 0.0139379, 0.0130971, 0.0139545, -0.0006412, 0.0006196
3: 1.0084695, 1.0090274, 1.0084658, 1.0090250, -0.0005547, 0.0005288
4: -0.0038625, -0.0037205, -0.0038656, -0.0037222, -0.0001000, 0.0001022
5: 0.0030942, 0.0035794, 0.0031033, 0.0035879, -0.0003779, 0.0003621
6: -0.0024351, -0.0023821, -0.0024333, -0.0023822, -0.0000529, 0.0000512
7: -0.0129394, -0.0121469, -0.0129406, -0.0122074, -0.0007173, 0.0007795
8: -0.0091715, -0.0076180, -0.0092075, -0.0076320, -0.0010886, 0.0010916
9: -0.0005802, 0.0001973, -0.0005766, 0.0002165, -0.0005355, 0.0005477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002855, upper bound: 0.0002780
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002888, upper bound: 0.0002808
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011097, -0.0004812, -0.0010833, -0.0004651, -0.0005052, 0.0004575
1: -0.0042254, -0.0040094, -0.0042228, -0.0040032, -0.0001732, 0.0001765
2: 0.0130838, 0.0139379, 0.0131129, 0.0139626, -0.0006603, 0.0006040
3: 1.0084695, 1.0090274, 1.0084625, 1.0090209, -0.0005515, 0.0005297
4: -0.0038625, -0.0037205, -0.0038671, -0.0037241, -0.0000989, 0.0001057
5: 0.0030942, 0.0035794, 0.0031139, 0.0035921, -0.0003877, 0.0003515
6: -0.0024351, -0.0023821, -0.0024330, -0.0023826, -0.0000525, 0.0000510
7: -0.0129394, -0.0121469, -0.0129413, -0.0122432, -0.0006810, 0.0007810
8: -0.0091715, -0.0076180, -0.0092250, -0.0076480, -0.0010831, 0.0011330
9: -0.0005802, 0.0001973, -0.0005702, 0.0002259, -0.0005576, 0.0005464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002855, upper bound: 0.0002780
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002888, upper bound: 0.0002808
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011097, -0.0004812, -0.0011097, -0.0004812, -0.0004800, 0.0004800
1: -0.0042254, -0.0040094, -0.0042254, -0.0040094, -0.0001753, 0.0001753
2: 0.0130838, 0.0139379, 0.0130838, 0.0139379, -0.0006239, 0.0006239
3: 1.0084695, 1.0090274, 1.0084695, 1.0090274, -0.0005569, 0.0005569
4: -0.0038625, -0.0037205, -0.0038625, -0.0037205, -0.0000993, 0.0000993
5: 0.0030942, 0.0035794, 0.0030942, 0.0035794, -0.0003681, 0.0003681
6: -0.0024351, -0.0023821, -0.0024351, -0.0023821, -0.0000531, 0.0000531
7: -0.0129394, -0.0121469, -0.0129394, -0.0121469, -0.0007781, 0.0007781
8: -0.0091715, -0.0076180, -0.0091715, -0.0076180, -0.0010729, 0.0010729
9: -0.0005802, 0.0001973, -0.0005802, 0.0001973, -0.0005365, 0.0005365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002924, upper bound: 0.0002910
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002921, upper bound: 0.0002910
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011097, -0.0004812, -0.0010987, -0.0004762, -0.0004921, 0.0004727
1: -0.0042254, -0.0040094, -0.0042236, -0.0040070, -0.0001793, 0.0001754
2: 0.0130838, 0.0139379, 0.0130964, 0.0139456, -0.0006425, 0.0006151
3: 1.0084695, 1.0090274, 1.0084643, 1.0090227, -0.0005533, 0.0005594
4: -0.0038625, -0.0037205, -0.0038639, -0.0037222, -0.0000986, 0.0001027
5: 0.0030942, 0.0035794, 0.0031025, 0.0035833, -0.0003776, 0.0003626
6: -0.0024351, -0.0023821, -0.0024351, -0.0023824, -0.0000527, 0.0000530
7: -0.0129394, -0.0121469, -0.0129400, -0.0121725, -0.0007527, 0.0007795
8: -0.0091715, -0.0076180, -0.0091881, -0.0076340, -0.0010709, 0.0011131
9: -0.0005802, 0.0001973, -0.0005734, 0.0002062, -0.0005580, 0.0005364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002908, upper bound: 0.0002910
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002921, upper bound: 0.0002910
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010816, -0.0004789, -0.0011036, -0.0004697, -0.0004691, 0.0004845
1: -0.0042211, -0.0040085, -0.0042257, -0.0040049, -0.0001673, 0.0001787
2: 0.0131165, 0.0139413, 0.0130903, 0.0139556, -0.0006159, 0.0006375
3: 1.0084692, 1.0090163, 1.0084643, 1.0090281, -0.0005445, 0.0005256
4: -0.0038631, -0.0037249, -0.0038658, -0.0037212, -0.0001034, 0.0000991
5: 0.0031154, 0.0035812, 0.0030987, 0.0035884, -0.0003602, 0.0003721
6: -0.0024340, -0.0023830, -0.0024337, -0.0023820, -0.0000520, 0.0000507
7: -0.0129396, -0.0122152, -0.0129407, -0.0121915, -0.0007340, 0.0007118
8: -0.0091789, -0.0076594, -0.0092097, -0.0076218, -0.0011258, 0.0010636
9: -0.0005638, 0.0002013, -0.0005809, 0.0002177, -0.0005256, 0.0005667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002725
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002725
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010816, -0.0004789, -0.0011170, -0.0004803, -0.0004558, 0.0004962
1: -0.0042211, -0.0040085, -0.0042266, -0.0040090, -0.0001718, 0.0001766
2: 0.0131165, 0.0139413, 0.0130755, 0.0139392, -0.0005969, 0.0006442
3: 1.0084692, 1.0090163, 1.0084674, 1.0090302, -0.0005458, 0.0005488
4: -0.0038631, -0.0037249, -0.0038627, -0.0037193, -0.0001028, 0.0000960
5: 0.0031154, 0.0035812, 0.0030886, 0.0035801, -0.0003500, 0.0003805
6: -0.0024340, -0.0023830, -0.0024357, -0.0023818, -0.0000522, 0.0000528
7: -0.0129396, -0.0122152, -0.0129395, -0.0121267, -0.0007997, 0.0007101
8: -0.0091789, -0.0076594, -0.0091743, -0.0076076, -0.0011119, 0.0010425
9: -0.0005638, 0.0002013, -0.0005844, 0.0001988, -0.0005218, 0.0005561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002871
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002871
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010814, -0.0004769, -0.0011039, -0.0004696, -0.0004684, 0.0004852
1: -0.0042208, -0.0040078, -0.0042257, -0.0040049, -0.0001673, 0.0001786
2: 0.0131161, 0.0139445, 0.0130900, 0.0139557, -0.0006148, 0.0006383
3: 1.0084671, 1.0090158, 1.0084642, 1.0090280, -0.0005456, 0.0005258
4: -0.0038637, -0.0037248, -0.0038658, -0.0037212, -0.0001035, 0.0000989
5: 0.0031155, 0.0035828, 0.0030985, 0.0035885, -0.0003597, 0.0003726
6: -0.0024340, -0.0023830, -0.0024337, -0.0023820, -0.0000520, 0.0000507
7: -0.0129399, -0.0122171, -0.0129407, -0.0121907, -0.0007348, 0.0007101
8: -0.0091858, -0.0076589, -0.0092100, -0.0076220, -0.0011277, 0.0010627
9: -0.0005629, 0.0002050, -0.0005808, 0.0002179, -0.0005255, 0.0005675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002761
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002761
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010814, -0.0004769, -0.0011170, -0.0004803, -0.0004550, 0.0004972
1: -0.0042208, -0.0040078, -0.0042265, -0.0040089, -0.0001719, 0.0001771
2: 0.0131161, 0.0139445, 0.0130755, 0.0139393, -0.0005960, 0.0006458
3: 1.0084671, 1.0090158, 1.0084674, 1.0090302, -0.0005471, 0.0005484
4: -0.0038637, -0.0037248, -0.0038628, -0.0037193, -0.0001031, 0.0000960
5: 0.0031155, 0.0035828, 0.0030886, 0.0035801, -0.0003493, 0.0003813
6: -0.0024340, -0.0023830, -0.0024357, -0.0023818, -0.0000522, 0.0000527
7: -0.0129399, -0.0122171, -0.0129395, -0.0121266, -0.0007999, 0.0007086
8: -0.0091858, -0.0076589, -0.0091745, -0.0076076, -0.0011154, 0.0010432
9: -0.0005629, 0.0002050, -0.0005843, 0.0001989, -0.0005221, 0.0005579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002873
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002873
time: 0.55 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.86 seconds
IS_A1_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002482, upper bound: 0.0002987
IS_A1_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002480, upper bound: 0.0002986
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0002711
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002986, upper bound: 0.0002711
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0003244
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002986, upper bound: 0.0003249
IS_A1_B1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002913, upper bound: 0.0002685
IS_A1_B1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003311, upper bound: 0.0003278
IS_A1_B1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002913, upper bound: 0.0002690
IS_A1_B1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003311, upper bound: 0.0003333
IS_A1_B1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002914, upper bound: 0.0002687
IS_A1_B1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003313, upper bound: 0.0003278
IS_A1_B1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002914, upper bound: 0.0002690
IS_A1_B1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003313, upper bound: 0.0003333
IS_A1_B1_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002685, upper bound: 0.0002913
IS_A1_B1_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003277, upper bound: 0.0003311
IS_A1_B1_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002690, upper bound: 0.0002913
IS_A1_B1_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003333, upper bound: 0.0003311
IS_A1_B1_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002687, upper bound: 0.0002914
IS_A1_B1_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003278, upper bound: 0.0003313
IS_A1_B1_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002690, upper bound: 0.0002914
IS_A1_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003333, upper bound: 0.0003313
IS_A1_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003270
IS_A1_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003319
IS_A1_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003270
IS_A1_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003319
IS_A1_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003272
IS_A1_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003321
IS_A1_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003272
IS_A1_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003331, upper bound: 0.0003320
IS_A1_B2_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002002, upper bound: 0.0002036
IS_A1_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003172, upper bound: 0.0003039
IS_A1_B2_A1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002096, upper bound: 0.0002111
IS_A1_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003178, upper bound: 0.0003038
IS_A1_B2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003066, upper bound: 0.0002940
IS_A1_B2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003066, upper bound: 0.0002943
IS_A1_B2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0002954
IS_A1_B2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0002966
IS_A1_B2_A1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002074, upper bound: 0.0002233
IS_A1_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003189, upper bound: 0.0003121
IS_A1_B2_A1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002124, upper bound: 0.0002241
IS_A1_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003190, upper bound: 0.0003130
IS_A1_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003088
IS_A1_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003095
IS_A1_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003088
IS_A1_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003118, upper bound: 0.0003095
IS_A1_B2_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002992, upper bound: 0.0003026
IS_A1_B2_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002992, upper bound: 0.0003026
IS_A1_B2_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002992, upper bound: 0.0003026
IS_A1_B2_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002992, upper bound: 0.0003026
IS_A1_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003046
IS_A1_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003046
IS_A1_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003046
IS_A1_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0003046
IS_A1_B2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002915
IS_A1_B2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002915
IS_A1_B2_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002992
IS_A1_B2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003085, upper bound: 0.0002992
IS_A1_B2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002930
IS_A1_B2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002930
IS_A1_B2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002996
IS_A1_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003096, upper bound: 0.0002996
IS_A2_B1_B1_A1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002036, upper bound: 0.0002002
IS_A2_B1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003039, upper bound: 0.0003172
IS_A2_B1_B1_A1_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002111, upper bound: 0.0002096
IS_A2_B1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003038, upper bound: 0.0003178
IS_A2_B1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002940, upper bound: 0.0003066
IS_A2_B1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002940, upper bound: 0.0003066
IS_A2_B1_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0003088
IS_A2_B1_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0003098
IS_A2_B1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002233, upper bound: 0.0002074
IS_A2_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003121, upper bound: 0.0003189
IS_A2_B1_B1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002241, upper bound: 0.0002124
IS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003130, upper bound: 0.0003190
IS_A2_B1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
IS_A2_B1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
IS_A2_B1_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
IS_A2_B1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003118
IS_A2_B1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0002992
IS_A2_B1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
IS_A2_B1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0002992
IS_A2_B1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
IS_A2_B1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
IS_A2_B1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003035
IS_A2_B1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003005
IS_A2_B1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0003026, upper bound: 0.0003035
IS_A2_B1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003085
IS_A2_B1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003085
IS_A2_B1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003079
IS_A2_B1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002915, upper bound: 0.0003079
IS_A2_B1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003096
IS_A2_B1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003096
IS_A2_B1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003080
IS_A2_B1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002930, upper bound: 0.0003080
IS_A2_B2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002758, upper bound: 0.0002921
IS_A2_B2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002758, upper bound: 0.0002921
IS_A2_B2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002758, upper bound: 0.0002921
IS_A2_B2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002758, upper bound: 0.0002921
IS_A2_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002944
IS_A2_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002945
IS_A2_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002944
IS_A2_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002945
IS_A2_B2_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002844
IS_A2_B2_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002844
IS_A2_B2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002844
IS_A2_B2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002844
IS_A2_B2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002876
IS_A2_B2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002878
IS_A2_B2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002876
IS_A2_B2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002878
IS_A2_B2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002855, upper bound: 0.0002780
IS_A2_B2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002888, upper bound: 0.0002808
IS_A2_B2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002855, upper bound: 0.0002780
IS_A2_B2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002888, upper bound: 0.0002808
IS_A2_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002924, upper bound: 0.0002910
IS_A2_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002921, upper bound: 0.0002910
IS_A2_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002908, upper bound: 0.0002910
IS_A2_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002921, upper bound: 0.0002910
IS_A2_B2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002725
IS_A2_B2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002725
IS_A2_B2_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002871
IS_A2_B2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002876, upper bound: 0.0002871
IS_A2_B2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002761
IS_A2_B2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002761
IS_A2_B2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002873
IS_A2_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.86
Output dim: 3, lower bound: -0.0002878, upper bound: 0.0002873

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011309, -0.0005053, -0.0010817, -0.0004577, -0.0005346, 0.0004463
1: -0.0042283, -0.0040200, -0.0042179, -0.0039983, -0.0001918, 0.0001609
2: 0.0130621, 0.0139008, 0.0131196, 0.0139739, -0.0006989, 0.0005798
3: 1.0084888, 1.0090346, 1.0084438, 1.0090085, -0.0005198, 0.0005841
4: -0.0038556, -0.0037177, -0.0038692, -0.0037261, -0.0000921, 0.0001129
5: 0.0030785, 0.0035604, 0.0031156, 0.0035978, -0.0004102, 0.0003422
6: -0.0024355, -0.0023815, -0.0024361, -0.0023836, -0.0000519, 0.0000547
7: -0.0129365, -0.0120913, -0.0129421, -0.0121967, -0.0007260, 0.0008355
8: -0.0090910, -0.0075918, -0.0092495, -0.0076765, -0.0009799, 0.0012239
9: -0.0005911, 0.0001543, -0.0005524, 0.0002390, -0.0006142, 0.0004798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002712, upper bound: 0.0002913
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002712, upper bound: 0.0002986
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011311, -0.0005053, -0.0010826, -0.0004569, -0.0005351, 0.0004491
1: -0.0042283, -0.0040199, -0.0042174, -0.0039973, -0.0001923, 0.0001615
2: 0.0130619, 0.0139009, 0.0131203, 0.0139752, -0.0006997, 0.0005818
3: 1.0084887, 1.0090346, 1.0084404, 1.0090072, -0.0005186, 0.0005868
4: -0.0038556, -0.0037177, -0.0038694, -0.0037264, -0.0000924, 0.0001131
5: 0.0030784, 0.0035604, 0.0031151, 0.0035985, -0.0004105, 0.0003442
6: -0.0024355, -0.0023815, -0.0024364, -0.0023837, -0.0000518, 0.0000549
7: -0.0129365, -0.0120911, -0.0129422, -0.0121910, -0.0007319, 0.0008357
8: -0.0090912, -0.0075919, -0.0092522, -0.0076804, -0.0009845, 0.0012251
9: -0.0005910, 0.0001544, -0.0005504, 0.0002404, -0.0006148, 0.0004819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002914
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002711, upper bound: 0.0002984
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010817, -0.0004577, -0.0011309, -0.0005053, -0.0004463, 0.0005346
1: -0.0042179, -0.0039983, -0.0042283, -0.0040200, -0.0001609, 0.0001918
2: 0.0131196, 0.0139739, 0.0130621, 0.0139008, -0.0005798, 0.0006989
3: 1.0084438, 1.0090085, 1.0084888, 1.0090346, -0.0005841, 0.0005198
4: -0.0038692, -0.0037261, -0.0038556, -0.0037177, -0.0001129, 0.0000921
5: 0.0031156, 0.0035978, 0.0030785, 0.0035604, -0.0003422, 0.0004102
6: -0.0024361, -0.0023836, -0.0024355, -0.0023815, -0.0000547, 0.0000519
7: -0.0129421, -0.0121967, -0.0129365, -0.0120913, -0.0008355, 0.0007260
8: -0.0092495, -0.0076765, -0.0090910, -0.0075918, -0.0012239, 0.0009799
9: -0.0005524, 0.0002390, -0.0005911, 0.0001543, -0.0004798, 0.0006142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002913, upper bound: 0.0002711
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002986, upper bound: 0.0002711
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010826, -0.0004569, -0.0011311, -0.0005053, -0.0004491, 0.0005351
1: -0.0042174, -0.0039973, -0.0042283, -0.0040199, -0.0001615, 0.0001923
2: 0.0131203, 0.0139752, 0.0130619, 0.0139009, -0.0005818, 0.0006997
3: 1.0084404, 1.0090072, 1.0084887, 1.0090346, -0.0005868, 0.0005186
4: -0.0038694, -0.0037264, -0.0038556, -0.0037177, -0.0001131, 0.0000924
5: 0.0031151, 0.0035985, 0.0030784, 0.0035604, -0.0003442, 0.0004105
6: -0.0024364, -0.0023837, -0.0024355, -0.0023815, -0.0000549, 0.0000518
7: -0.0129422, -0.0121910, -0.0129365, -0.0120911, -0.0008357, 0.0007319
8: -0.0092522, -0.0076804, -0.0090912, -0.0075919, -0.0012251, 0.0009845
9: -0.0005504, 0.0002404, -0.0005910, 0.0001544, -0.0004819, 0.0006148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002914, upper bound: 0.0002711
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002984, upper bound: 0.0002711
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010817, -0.0004577, -0.0010983, -0.0004551, -0.0004888, 0.0005018
1: -0.0042179, -0.0039983, -0.0042202, -0.0039969, -0.0001873, 0.0001859
2: 0.0131196, 0.0139739, 0.0131019, 0.0139780, -0.0006468, 0.0006585
3: 1.0084438, 1.0090085, 1.0084397, 1.0090144, -0.0005705, 0.0005689
4: -0.0038692, -0.0037261, -0.0038700, -0.0037237, -0.0001073, 0.0001062
5: 0.0031156, 0.0035978, 0.0031032, 0.0035999, -0.0003755, 0.0003851
6: -0.0024361, -0.0023836, -0.0024372, -0.0023831, -0.0000530, 0.0000536
7: -0.0129421, -0.0121967, -0.0129424, -0.0121518, -0.0007752, 0.0007308
8: -0.0092495, -0.0076765, -0.0092583, -0.0076546, -0.0011688, 0.0011609
9: -0.0005524, 0.0002390, -0.0005610, 0.0002437, -0.0005868, 0.0005894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003285, upper bound: 0.0003244
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003339, upper bound: 0.0003243
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010826, -0.0004569, -0.0010983, -0.0004550, -0.0004900, 0.0005013
1: -0.0042174, -0.0039973, -0.0042201, -0.0039969, -0.0001880, 0.0001865
2: 0.0131203, 0.0139752, 0.0131020, 0.0139781, -0.0006474, 0.0006579
3: 1.0084404, 1.0090072, 1.0084397, 1.0090141, -0.0005738, 0.0005676
4: -0.0038694, -0.0037264, -0.0038700, -0.0037238, -0.0001071, 0.0001063
5: 0.0031151, 0.0035985, 0.0031032, 0.0036000, -0.0003764, 0.0003847
6: -0.0024364, -0.0023837, -0.0024372, -0.0023831, -0.0000533, 0.0000535
7: -0.0129422, -0.0121910, -0.0129424, -0.0121516, -0.0007753, 0.0007365
8: -0.0092522, -0.0076804, -0.0092585, -0.0076550, -0.0011667, 0.0011639
9: -0.0005504, 0.0002404, -0.0005609, 0.0002438, -0.0005891, 0.0005883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.01 + 598.23 = 601.25 seconds
