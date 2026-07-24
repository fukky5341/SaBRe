## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000623675


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0018171, -0.0010423, -0.0018171, -0.0010423, -0.0004235, 0.0004235)
1: (-0.0089218, -0.0069555, -0.0089218, -0.0069555, -0.0010748, 0.0010748)
2: (0.0294949, 0.0307148, 0.0294949, 0.0307148, -0.0006668, 0.0006668)
3: (0.0022102, 0.0044880, 0.0022102, 0.0044880, -0.0012451, 0.0012451)
4: (-0.0079680, -0.0059680, -0.0079680, -0.0059680, -0.0010932, 0.0010932)
5: (0.0107201, 0.0114777, 0.0107201, 0.0114777, -0.0004141, 0.0004141)
6: (0.0031932, 0.0060840, 0.0031932, 0.0060840, -0.0015802, 0.0015802)
7: (0.9802938, 0.9823166, 0.9802938, 0.9823166, -0.0011057, 0.0011057)
8: (-0.0076925, -0.0055237, -0.0076925, -0.0055237, -0.0011855, 0.0011855)
9: (-0.0013509, 0.0000817, -0.0013509, 0.0000817, -0.0007831, 0.0007831)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 1.69 = 3.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0006701, upper bound: 0.0006701

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0006436, upper bound: 0.0006316
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0006438, upper bound: 0.0006438
time: 0.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 7, lower bound: -0.0006436, upper bound: 0.0006316
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 7, lower bound: -0.0006438, upper bound: 0.0006438

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0017759, -0.0010455, -0.0017996, -0.0010427, -0.0003676, 0.0003732
1: -0.0088172, -0.0069636, -0.0088774, -0.0069566, -0.0009329, 0.0009471
2: 0.0295598, 0.0307098, 0.0295224, 0.0307141, -0.0005788, 0.0005876
3: 0.0022195, 0.0043668, 0.0022115, 0.0044366, -0.0010972, 0.0010808
4: -0.0078616, -0.0059761, -0.0079228, -0.0059691, -0.0009490, 0.0009633
5: 0.0107604, 0.0114746, 0.0107372, 0.0114773, -0.0003594, 0.0003649
6: 0.0032050, 0.0059302, 0.0031948, 0.0060188, -0.0013924, 0.0013716
7: 0.9803019, 0.9822090, 0.9802948, 0.9822709, -0.0009744, 0.0009598
8: -0.0076836, -0.0056390, -0.0076913, -0.0055726, -0.0010447, 0.0010291
9: -0.0012747, 0.0000759, -0.0013186, 0.0000809, -0.0006797, 0.0006901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006194, upper bound: 0.0006093
time: 0.83 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006200, upper bound: 0.0006093
time: 0.78 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0017999, -0.0010429, -0.0018110, -0.0010425, -0.0003496, 0.0004202
1: -0.0088781, -0.0069570, -0.0089061, -0.0069561, -0.0008871, 0.0010663
2: 0.0295220, 0.0307138, 0.0295046, 0.0307145, -0.0005504, 0.0006616
3: 0.0022120, 0.0044374, 0.0022108, 0.0044699, -0.0012353, 0.0010277
4: -0.0079235, -0.0059695, -0.0079521, -0.0059685, -0.0009024, 0.0010846
5: 0.0107370, 0.0114771, 0.0107262, 0.0114775, -0.0003418, 0.0004108
6: 0.0031954, 0.0060198, 0.0031940, 0.0060610, -0.0015677, 0.0013043
7: 0.9802952, 0.9822716, 0.9802943, 0.9823004, -0.0010970, 0.0009127
8: -0.0076908, -0.0055718, -0.0076919, -0.0055409, -0.0011762, 0.0009785
9: -0.0013191, 0.0000806, -0.0013395, 0.0000813, -0.0006464, 0.0007769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006197, upper bound: 0.0006203
time: 0.81 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006203, upper bound: 0.0006203
time: 0.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.00 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 7, lower bound: -0.0006194, upper bound: 0.0006093
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 7, lower bound: -0.0006200, upper bound: 0.0006093
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 7, lower bound: -0.0006197, upper bound: 0.0006203
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 7, lower bound: -0.0006203, upper bound: 0.0006203

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.06 + 7.73 = 10.78 seconds
