## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00085992


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0068952, 0.0083904, 0.0068952, 0.0083904, -0.0009989, 0.0009989)
1: (0.0023185, 0.0025345, 0.0023185, 0.0025345, -0.0001443, 0.0001443)
2: (0.0097210, 0.0105477, 0.0097210, 0.0105477, -0.0005523, 0.0005523)
3: (-0.0046265, -0.0037716, -0.0046265, -0.0037716, -0.0005712, 0.0005712)
4: (0.0000460, 0.0009715, 0.0000460, 0.0009715, -0.0006183, 0.0006183)
5: (0.0031940, 0.0040698, 0.0031940, 0.0040698, -0.0005851, 0.0005851)
6: (-0.0096276, -0.0061524, -0.0096276, -0.0061524, -0.0023217, 0.0023217)
7: (0.0058223, 0.0105553, 0.0058223, 0.0105553, -0.0031619, 0.0031619)
8: (0.9933152, 0.9966492, 0.9933152, 0.9966492, -0.0022273, 0.0022273)
9: (-0.0128457, -0.0098193, -0.0128457, -0.0098193, -0.0020218, 0.0020218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.82 + 1.30 = 3.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0009649, upper bound: 0.0009650

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008073, upper bound: 0.0008875
time: 0.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009009, upper bound: 0.0009010
time: 0.46 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 8, lower bound: -0.0008073, upper bound: 0.0008875
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 8, lower bound: -0.0009009, upper bound: 0.0009010

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0071458, 0.0084203, 0.0070087, 0.0083902, -0.0007028, 0.0008192
1: 0.0023547, 0.0025388, 0.0023349, 0.0025344, -0.0001015, 0.0001184
2: 0.0097045, 0.0104091, 0.0097211, 0.0104849, -0.0004529, 0.0003885
3: -0.0046436, -0.0039148, -0.0046264, -0.0038364, -0.0004684, 0.0004019
4: 0.0002011, 0.0009900, 0.0001162, 0.0009714, -0.0004350, 0.0005071
5: 0.0031764, 0.0039231, 0.0031941, 0.0040034, -0.0004799, 0.0004117
6: -0.0096972, -0.0067348, -0.0096271, -0.0064161, -0.0019040, 0.0016334
7: 0.0066155, 0.0106500, 0.0061815, 0.0105546, -0.0022246, 0.0025931
8: 0.9938740, 0.9967160, 0.9935683, 0.9966487, -0.0015671, 0.0018267
9: -0.0129062, -0.0103264, -0.0128452, -0.0100490, -0.0016581, 0.0014225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007673, upper bound: 0.0008207
time: 0.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007727, upper bound: 0.0008444
time: 0.49 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0069350, 0.0083903, 0.0069055, 0.0083904, -0.0007134, 0.0009923
1: 0.0023242, 0.0025345, 0.0023199, 0.0025345, -0.0001031, 0.0001434
2: 0.0097211, 0.0105257, 0.0097210, 0.0105420, -0.0005486, 0.0003944
3: -0.0046264, -0.0037943, -0.0046265, -0.0037774, -0.0005674, 0.0004079
4: 0.0000706, 0.0009714, 0.0000523, 0.0009715, -0.0004416, 0.0006142
5: 0.0031940, 0.0040465, 0.0031940, 0.0040638, -0.0005813, 0.0004179
6: -0.0096273, -0.0062449, -0.0096275, -0.0061763, -0.0023063, 0.0016581
7: 0.0059482, 0.0105548, 0.0058549, 0.0105552, -0.0022582, 0.0031410
8: 0.9934039, 0.9966490, 0.9933382, 0.9966493, -0.0015907, 0.0022126
9: -0.0128454, -0.0098998, -0.0128456, -0.0098401, -0.0020084, 0.0014440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008488, upper bound: 0.0008312
time: 0.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008590, upper bound: 0.0008590
time: 0.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.73 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 8, lower bound: -0.0007673, upper bound: 0.0008207
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 8, lower bound: -0.0007727, upper bound: 0.0008444
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 8, lower bound: -0.0008488, upper bound: 0.0008312
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 8, lower bound: -0.0008590, upper bound: 0.0008590

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.11 + 6.68 = 9.79 seconds
