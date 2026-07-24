## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00014742


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0060802, 0.0063387, 0.0060802, 0.0063387, -0.0001185, 0.0001185)
1: (-0.0001412, 0.0003596, -0.0001412, 0.0003596, -0.0002296, 0.0002296)
2: (0.0140278, 0.0180665, 0.0140278, 0.0180665, -0.0018516, 0.0018516)
3: (-0.0041262, -0.0037655, -0.0041262, -0.0037655, -0.0001654, 0.0001654)
4: (0.0013167, 0.0030668, 0.0013167, 0.0030668, -0.0008024, 0.0008024)
5: (-0.0009776, -0.0007163, -0.0009776, -0.0007163, -0.0001198, 0.0001198)
6: (0.9915668, 0.9920460, 0.9915668, 0.9920460, -0.0002197, 0.0002197)
7: (-0.0109994, -0.0078313, -0.0109994, -0.0078313, -0.0014525, 0.0014525)
8: (-0.0024577, -0.0014652, -0.0024577, -0.0014652, -0.0004550, 0.0004550)
9: (-0.0044049, -0.0024239, -0.0044049, -0.0024239, -0.0009082, 0.0009082)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 1.44 = 2.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0001692, upper bound: 0.0001692

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 95

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001576, upper bound: 0.0001583
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001583
time: 0.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.19 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 6, lower bound: -0.0001576, upper bound: 0.0001583
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001583

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0060875, 0.0063386, 0.0060811, 0.0063387, -0.0001091, 0.0001170
1: -0.0001269, 0.0003594, -0.0001394, 0.0003595, -0.0002113, 0.0002267
2: 0.0140294, 0.0179513, 0.0140280, 0.0180520, -0.0018282, 0.0017041
3: -0.0041159, -0.0037656, -0.0041249, -0.0037655, -0.0001522, 0.0001633
4: 0.0013666, 0.0030661, 0.0013230, 0.0030668, -0.0007385, 0.0007922
5: -0.0009775, -0.0007237, -0.0009775, -0.0007172, -0.0001183, 0.0001102
6: 0.9915805, 0.9920458, 0.9915686, 0.9920459, -0.0002022, 0.0002169
7: -0.0109090, -0.0078326, -0.0109880, -0.0078315, -0.0013368, 0.0014340
8: -0.0024294, -0.0014656, -0.0024541, -0.0014652, -0.0004188, 0.0004493
9: -0.0044041, -0.0024805, -0.0044048, -0.0024310, -0.0008967, 0.0008359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 95

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001575, upper bound: 0.0001575
time: 0.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001575, upper bound: 0.0001583
time: 0.58 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0060961, 0.0063498, 0.0060865, 0.0063386, -0.0001097, 0.0001461
1: -0.0001103, 0.0003811, -0.0001288, 0.0003593, -0.0002125, 0.0002830
2: 0.0138541, 0.0178175, 0.0140297, 0.0179671, -0.0022827, 0.0017137
3: -0.0041039, -0.0037499, -0.0041173, -0.0037656, -0.0001531, 0.0002039
4: 0.0014246, 0.0031421, 0.0013598, 0.0030660, -0.0007426, 0.0009892
5: -0.0009888, -0.0007324, -0.0009774, -0.0007227, -0.0001477, 0.0001109
6: 0.9915963, 0.9920666, 0.9915786, 0.9920457, -0.0002033, 0.0002708
7: -0.0108041, -0.0076951, -0.0109214, -0.0078329, -0.0013442, 0.0017906
8: -0.0023965, -0.0014225, -0.0024332, -0.0014656, -0.0004211, 0.0005610
9: -0.0044901, -0.0025461, -0.0044039, -0.0024727, -0.0011197, 0.0008405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 95

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001575
time: 0.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001583
time: 0.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.68 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 6, lower bound: -0.0001575, upper bound: 0.0001575
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 6, lower bound: -0.0001575, upper bound: 0.0001583
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001575
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001583

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0060875, 0.0063386, 0.0060875, 0.0063386, -0.0001090, 0.0001090
1: -0.0001269, 0.0003594, -0.0001269, 0.0003594, -0.0002112, 0.0002112
2: 0.0140294, 0.0179513, 0.0140294, 0.0179513, -0.0017032, 0.0017031
3: -0.0041159, -0.0037656, -0.0041159, -0.0037656, -0.0001521, 0.0001521
4: 0.0013666, 0.0030661, 0.0013666, 0.0030661, -0.0007380, 0.0007380
5: -0.0009775, -0.0007237, -0.0009775, -0.0007237, -0.0001102, 0.0001102
6: 0.9915805, 0.9920458, 0.9915805, 0.9920458, -0.0002021, 0.0002021
7: -0.0109090, -0.0078326, -0.0109090, -0.0078326, -0.0013360, 0.0013360
8: -0.0024294, -0.0014656, -0.0024294, -0.0014656, -0.0004186, 0.0004186
9: -0.0044041, -0.0024805, -0.0044041, -0.0024805, -0.0008354, 0.0008354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 95

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001471, upper bound: 0.0001472
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001474, upper bound: 0.0001466
time: 0.53 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0060875, 0.0063386, 0.0060961, 0.0063498, -0.0001372, 0.0001150
1: -0.0001269, 0.0003594, -0.0001103, 0.0003811, -0.0002657, 0.0002228
2: 0.0140294, 0.0179513, 0.0138541, 0.0178175, -0.0017972, 0.0021433
3: -0.0041159, -0.0037656, -0.0041039, -0.0037499, -0.0001914, 0.0001605
4: 0.0013666, 0.0030661, 0.0014246, 0.0031421, -0.0009288, 0.0007788
5: -0.0009775, -0.0007237, -0.0009888, -0.0007324, -0.0001163, 0.0001386
6: 0.9915805, 0.9920458, 0.9915963, 0.9920666, -0.0002543, 0.0002132
7: -0.0109090, -0.0078326, -0.0108041, -0.0076951, -0.0016813, 0.0014098
8: -0.0024294, -0.0014656, -0.0023965, -0.0014225, -0.0005267, 0.0004417
9: -0.0044041, -0.0024805, -0.0044901, -0.0025461, -0.0008815, 0.0010513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 95

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001471, upper bound: 0.0001473
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001474, upper bound: 0.0001466
time: 0.54 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0060961, 0.0063498, 0.0060875, 0.0063386, -0.0001150, 0.0001372
1: -0.0001103, 0.0003811, -0.0001269, 0.0003594, -0.0002228, 0.0002657
2: 0.0138541, 0.0178175, 0.0140294, 0.0179513, -0.0021433, 0.0017972
3: -0.0041039, -0.0037499, -0.0041159, -0.0037656, -0.0001605, 0.0001914
4: 0.0014246, 0.0031421, 0.0013666, 0.0030661, -0.0007788, 0.0009288
5: -0.0009888, -0.0007324, -0.0009775, -0.0007237, -0.0001386, 0.0001163
6: 0.9915963, 0.9920666, 0.9915805, 0.9920458, -0.0002132, 0.0002543
7: -0.0108041, -0.0076951, -0.0109090, -0.0078326, -0.0014098, 0.0016813
8: -0.0023965, -0.0014225, -0.0024294, -0.0014656, -0.0004417, 0.0005267
9: -0.0044901, -0.0025461, -0.0044041, -0.0024805, -0.0010513, 0.0008815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 95

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001470, upper bound: 0.0001471
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001466, upper bound: 0.0001465
time: 0.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0060961, 0.0063498, 0.0060961, 0.0063498, -0.0001082, 0.0001082
1: -0.0001103, 0.0003811, -0.0001103, 0.0003811, -0.0002096, 0.0002096
2: 0.0138541, 0.0178175, 0.0138541, 0.0178175, -0.0016902, 0.0016902
3: -0.0041039, -0.0037499, -0.0041039, -0.0037499, -0.0001510, 0.0001510
4: 0.0014246, 0.0031421, 0.0014246, 0.0031421, -0.0007325, 0.0007325
5: -0.0009888, -0.0007324, -0.0009888, -0.0007324, -0.0001093, 0.0001093
6: 0.9915963, 0.9920666, 0.9915963, 0.9920666, -0.0002005, 0.0002005
7: -0.0108041, -0.0076951, -0.0108041, -0.0076951, -0.0013259, 0.0013259
8: -0.0023965, -0.0014225, -0.0023965, -0.0014225, -0.0004154, 0.0004154
9: -0.0044901, -0.0025461, -0.0044901, -0.0025461, -0.0008290, 0.0008290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 95

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001470, upper bound: 0.0001471
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001466, upper bound: 0.0001465
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.72 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 6, lower bound: -0.0001471, upper bound: 0.0001472
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 6, lower bound: -0.0001474, upper bound: 0.0001466
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 6, lower bound: -0.0001471, upper bound: 0.0001473
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 6, lower bound: -0.0001474, upper bound: 0.0001466
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 6, lower bound: -0.0001470, upper bound: 0.0001471
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 6, lower bound: -0.0001466, upper bound: 0.0001465
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 6, lower bound: -0.0001470, upper bound: 0.0001471
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 6, lower bound: -0.0001466, upper bound: 0.0001465

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.96 + 17.24 = 20.20 seconds
