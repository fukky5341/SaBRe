## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.004636575


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002972, 0.0001315, -0.0002972, 0.0001315, -0.0003368, 0.0003368)
1: (-0.0000711, 0.0015232, -0.0000711, 0.0015232, -0.0012734, 0.0012734)
2: (0.0140588, 0.0164464, 0.0140588, 0.0164464, -0.0018858, 0.0018858)
3: (-0.0000553, 0.0017401, -0.0000553, 0.0017401, -0.0014088, 0.0014088)
4: (-0.0044307, -0.0027745, -0.0044307, -0.0027745, -0.0013816, 0.0013816)
5: (0.0078830, 0.0096751, 0.0078830, 0.0096751, -0.0014054, 0.0014054)
6: (0.0092818, 0.0099581, 0.0092818, 0.0099581, -0.0006763, 0.0006763)
7: (-0.0194031, -0.0155125, -0.0194031, -0.0155125, -0.0029623, 0.0029623)
8: (0.9681987, 0.9793457, 0.9681987, 0.9793457, -0.0088259, 0.0088259)
9: (0.0037235, 0.0069996, 0.0037235, 0.0069996, -0.0025201, 0.0025201)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.50 = 2.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0062004, upper bound: 0.0062004

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056317, upper bound: 0.0059143
time: 0.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059143, upper bound: 0.0059143
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 8, lower bound: -0.0056317, upper bound: 0.0059143
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 8, lower bound: -0.0059143, upper bound: 0.0059143

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0002702, 0.0001190, -0.0002935, 0.0001313, -0.0003072, 0.0003147
1: 0.0000554, 0.0015040, -0.0000536, 0.0015229, -0.0011415, 0.0012049
2: 0.0140875, 0.0162570, 0.0140593, 0.0164202, -0.0017830, 0.0016869
3: -0.0000337, 0.0015977, -0.0000549, 0.0017204, -0.0013315, 0.0012587
4: -0.0044107, -0.0029059, -0.0044303, -0.0027927, -0.0013110, 0.0012444
5: 0.0079045, 0.0095329, 0.0078834, 0.0096554, -0.0013282, 0.0012555
6: 0.0093355, 0.0099499, 0.0092892, 0.0099579, -0.0006225, 0.0006607
7: -0.0190944, -0.0155594, -0.0193603, -0.0155134, -0.0026276, 0.0027941
8: 0.9690833, 0.9792114, 0.9683212, 0.9793432, -0.0078984, 0.0083455
9: 0.0037629, 0.0067396, 0.0037242, 0.0069636, -0.0023784, 0.0022413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056317, upper bound: 0.0056317
time: 0.63 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056317, upper bound: 0.0059143
time: 0.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0002888, 0.0001309, -0.0002972, 0.0001315, -0.0003072, 0.0003360
1: -0.0000316, 0.0015222, -0.0000711, 0.0015232, -0.0011504, 0.0012724
2: 0.0140604, 0.0163873, 0.0140588, 0.0164464, -0.0018844, 0.0016967
3: -0.0000541, 0.0016956, -0.0000553, 0.0017401, -0.0014077, 0.0012643
4: -0.0044295, -0.0028156, -0.0044307, -0.0027745, -0.0013806, 0.0012625
5: 0.0078842, 0.0096307, 0.0078830, 0.0096751, -0.0014043, 0.0012610
6: 0.0092986, 0.0099576, 0.0092818, 0.0099581, -0.0006595, 0.0006758
7: -0.0193067, -0.0155152, -0.0194031, -0.0155125, -0.0026255, 0.0029599
8: 0.9684749, 0.9793380, 0.9681987, 0.9793457, -0.0079483, 0.0088191
9: 0.0037257, 0.0069184, 0.0037235, 0.0069996, -0.0025181, 0.0022438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059143, upper bound: 0.0056317
time: 0.61 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059143, upper bound: 0.0059143
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 8, lower bound: -0.0056317, upper bound: 0.0056317
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 8, lower bound: -0.0056317, upper bound: 0.0059143
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 8, lower bound: -0.0059143, upper bound: 0.0056317
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 8, lower bound: -0.0059143, upper bound: 0.0059143

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002702, 0.0001190, -0.0002702, 0.0001190, -0.0002898, 0.0002898
1: 0.0000554, 0.0015040, 0.0000554, 0.0015040, -0.0010939, 0.0010939
2: 0.0140875, 0.0162570, 0.0140875, 0.0162570, -0.0016156, 0.0016156
3: -0.0000337, 0.0015977, -0.0000337, 0.0015977, -0.0012051, 0.0012051
4: -0.0044107, -0.0029059, -0.0044107, -0.0029059, -0.0011949, 0.0011949
5: 0.0079045, 0.0095329, 0.0079045, 0.0095329, -0.0012020, 0.0012020
6: 0.0093355, 0.0099499, 0.0093355, 0.0099499, -0.0006145, 0.0006145
7: -0.0190944, -0.0155594, -0.0190944, -0.0155594, -0.0025113, 0.0025113
8: 0.9690833, 0.9792114, 0.9690833, 0.9792114, -0.0075654, 0.0075654
9: 0.0037629, 0.0067396, 0.0037629, 0.0067396, -0.0021434, 0.0021434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054090, upper bound: 0.0052622
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054672, upper bound: 0.0054807
time: 0.61 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002702, 0.0001190, -0.0002888, 0.0001309, -0.0003067, 0.0003136
1: 0.0000554, 0.0015040, -0.0000316, 0.0015222, -0.0011408, 0.0011964
2: 0.0140875, 0.0162570, 0.0140604, 0.0163873, -0.0017734, 0.0016859
3: -0.0000337, 0.0015977, -0.0000541, 0.0016956, -0.0013251, 0.0012580
4: -0.0044107, -0.0029059, -0.0044295, -0.0028156, -0.0012943, 0.0012437
5: 0.0079045, 0.0095329, 0.0078842, 0.0096307, -0.0013219, 0.0012548
6: 0.0093355, 0.0099499, 0.0092986, 0.0099576, -0.0006222, 0.0006514
7: -0.0190944, -0.0155594, -0.0193067, -0.0155152, -0.0026260, 0.0027818
8: 0.9690833, 0.9792114, 0.9684749, 0.9793380, -0.0078938, 0.0082982
9: 0.0037629, 0.0067396, 0.0037257, 0.0069184, -0.0023683, 0.0022399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054090, upper bound: 0.0055401
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054672, upper bound: 0.0057467
time: 0.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002888, 0.0001309, -0.0002702, 0.0001190, -0.0003136, 0.0003067
1: -0.0000316, 0.0015222, 0.0000554, 0.0015040, -0.0011964, 0.0011408
2: 0.0140604, 0.0163873, 0.0140875, 0.0162570, -0.0016859, 0.0017734
3: -0.0000541, 0.0016956, -0.0000337, 0.0015977, -0.0012580, 0.0013251
4: -0.0044295, -0.0028156, -0.0044107, -0.0029059, -0.0012437, 0.0012943
5: 0.0078842, 0.0096307, 0.0079045, 0.0095329, -0.0012548, 0.0013219
6: 0.0092986, 0.0099576, 0.0093355, 0.0099499, -0.0006514, 0.0006222
7: -0.0193067, -0.0155152, -0.0190944, -0.0155594, -0.0027818, 0.0026260
8: 0.9684749, 0.9793380, 0.9690833, 0.9792114, -0.0082982, 0.0078938
9: 0.0037257, 0.0069184, 0.0037629, 0.0067396, -0.0022399, 0.0023683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056821, upper bound: 0.0052622
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057467, upper bound: 0.0054672
time: 0.66 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002888, 0.0001309, -0.0002888, 0.0001309, -0.0003064, 0.0003064
1: -0.0000316, 0.0015222, -0.0000316, 0.0015222, -0.0011493, 0.0011493
2: 0.0140604, 0.0163873, 0.0140604, 0.0163873, -0.0016951, 0.0016951
3: -0.0000541, 0.0016956, -0.0000541, 0.0016956, -0.0012631, 0.0012631
4: -0.0044295, -0.0028156, -0.0044295, -0.0028156, -0.0012613, 0.0012613
5: 0.0078842, 0.0096307, 0.0078842, 0.0096307, -0.0012597, 0.0012597
6: 0.0092986, 0.0099576, 0.0092986, 0.0099576, -0.0006591, 0.0006591
7: -0.0193067, -0.0155152, -0.0193067, -0.0155152, -0.0026228, 0.0026228
8: 0.9684749, 0.9793380, 0.9684749, 0.9793380, -0.0079404, 0.0079404
9: 0.0037257, 0.0069184, 0.0037257, 0.0069184, -0.0022415, 0.0022415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056821, upper bound: 0.0052829
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057467, upper bound: 0.0054672
time: 0.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 8, lower bound: -0.0054090, upper bound: 0.0052622
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 8, lower bound: -0.0054672, upper bound: 0.0054807
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 8, lower bound: -0.0054090, upper bound: 0.0055401
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 8, lower bound: -0.0054672, upper bound: 0.0057467
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 8, lower bound: -0.0056821, upper bound: 0.0052622
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 8, lower bound: -0.0057467, upper bound: 0.0054672
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 8, lower bound: -0.0056821, upper bound: 0.0052829
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 8, lower bound: -0.0057467, upper bound: 0.0054672

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000558, -0.0002699, 0.0001081, -0.0002785, 0.0002239
1: 0.0000121, 0.0014072, 0.0000568, 0.0014873, -0.0010284, 0.0009613
2: 0.0142326, 0.0163219, 0.0141126, 0.0162549, -0.0014198, 0.0015325
3: 0.0000754, 0.0016465, -0.0000148, 0.0015961, -0.0010590, 0.0011490
4: -0.0043101, -0.0028609, -0.0043933, -0.0029074, -0.0010504, 0.0010890
5: 0.0080135, 0.0095817, 0.0079234, 0.0095313, -0.0010562, 0.0011466
6: 0.0093171, 0.0099088, 0.0093360, 0.0099428, -0.0005343, 0.0005728
7: -0.0192002, -0.0157958, -0.0190910, -0.0156003, -0.0024502, 0.0022061
8: 0.9687800, 0.9785340, 0.9690930, 0.9790943, -0.0071623, 0.0066483
9: 0.0039620, 0.0068287, 0.0037973, 0.0067367, -0.0018832, 0.0020762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051559, upper bound: 0.0050322
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052356, upper bound: 0.0050322
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001025, -0.0002702, 0.0001190, -0.0002893, 0.0002612
1: 0.0000571, 0.0014787, 0.0000554, 0.0015040, -0.0010795, 0.0010062
2: 0.0141255, 0.0162544, 0.0140875, 0.0162570, -0.0014822, 0.0015981
3: -0.0000051, 0.0015958, -0.0000337, 0.0015977, -0.0011038, 0.0011935
4: -0.0043844, -0.0029077, -0.0044107, -0.0029059, -0.0011105, 0.0011698
5: 0.0079330, 0.0095310, 0.0079045, 0.0095329, -0.0011008, 0.0011905
6: 0.0093362, 0.0099392, 0.0093355, 0.0099499, -0.0006138, 0.0006037
7: -0.0190902, -0.0156212, -0.0190944, -0.0155594, -0.0025031, 0.0022916
8: 0.9690952, 0.9790342, 0.9690833, 0.9792114, -0.0074796, 0.0069445
9: 0.0038150, 0.0067361, 0.0037629, 0.0067396, -0.0019580, 0.0021318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052622, upper bound: 0.0054135
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052622, upper bound: 0.0054807
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000558, -0.0002884, 0.0001201, -0.0002961, 0.0002478
1: 0.0000121, 0.0014072, -0.0000302, 0.0015057, -0.0010777, 0.0010665
2: 0.0142326, 0.0163219, 0.0140850, 0.0163851, -0.0015803, 0.0016064
3: 0.0000754, 0.0016465, -0.0000356, 0.0016940, -0.0011805, 0.0012046
4: -0.0043101, -0.0028609, -0.0044125, -0.0028171, -0.0011529, 0.0011402
5: 0.0080135, 0.0095817, 0.0079026, 0.0096291, -0.0011776, 0.0012020
6: 0.0093171, 0.0099088, 0.0092992, 0.0099507, -0.0005552, 0.0006097
7: -0.0192002, -0.0157958, -0.0193032, -0.0155553, -0.0025705, 0.0024776
8: 0.9687800, 0.9785340, 0.9684850, 0.9792233, -0.0075070, 0.0073953
9: 0.0039620, 0.0068287, 0.0037594, 0.0069154, -0.0021094, 0.0021775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051545, upper bound: 0.0053090
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0053089
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001025, -0.0002888, 0.0001309, -0.0003062, 0.0002882
1: 0.0000571, 0.0014787, -0.0000316, 0.0015222, -0.0011264, 0.0011154
2: 0.0141255, 0.0162544, 0.0140604, 0.0163873, -0.0016518, 0.0016685
3: -0.0000051, 0.0015958, -0.0000541, 0.0016956, -0.0012337, 0.0012464
4: -0.0043844, -0.0029077, -0.0044295, -0.0028156, -0.0012117, 0.0012185
5: 0.0079330, 0.0095310, 0.0078842, 0.0096307, -0.0012306, 0.0012433
6: 0.0093362, 0.0099392, 0.0092986, 0.0099576, -0.0006215, 0.0006406
7: -0.0190902, -0.0156212, -0.0193067, -0.0155152, -0.0026177, 0.0025964
8: 0.9690952, 0.9790342, 0.9684749, 0.9793380, -0.0078080, 0.0077308
9: 0.0038150, 0.0067361, 0.0037257, 0.0069184, -0.0022093, 0.0022283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052622, upper bound: 0.0056821
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052622, upper bound: 0.0057467
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002978, 0.0000650, -0.0002699, 0.0001081, -0.0003067, 0.0002407
1: -0.0000738, 0.0014213, 0.0000568, 0.0014873, -0.0011560, 0.0010143
2: 0.0142114, 0.0164504, 0.0141126, 0.0162549, -0.0014991, 0.0017251
3: 0.0000595, 0.0017431, -0.0000148, 0.0015961, -0.0011186, 0.0012945
4: -0.0043248, -0.0027718, -0.0043933, -0.0029074, -0.0011054, 0.0012171
5: 0.0079976, 0.0096781, 0.0079234, 0.0095313, -0.0011157, 0.0012919
6: 0.0092807, 0.0099149, 0.0093360, 0.0099428, -0.0005787, 0.0005788
7: -0.0194096, -0.0157613, -0.0190910, -0.0156003, -0.0027714, 0.0023353
8: 0.9681801, 0.9786329, 0.9690930, 0.9790943, -0.0080601, 0.0070185
9: 0.0039329, 0.0070050, 0.0037973, 0.0067367, -0.0019920, 0.0023448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050322
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050322
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002884, 0.0001134, -0.0002702, 0.0001190, -0.0003132, 0.0002788
1: -0.0000300, 0.0014954, 0.0000554, 0.0015040, -0.0011853, 0.0010621
2: 0.0141005, 0.0163849, 0.0140875, 0.0162570, -0.0015659, 0.0017592
3: -0.0000239, 0.0016938, -0.0000337, 0.0015977, -0.0011667, 0.0013154
4: -0.0044017, -0.0028172, -0.0044107, -0.0029059, -0.0011685, 0.0012726
5: 0.0079143, 0.0096289, 0.0079045, 0.0095329, -0.0011636, 0.0013123
6: 0.0092992, 0.0099463, 0.0093355, 0.0099499, -0.0006507, 0.0006108
7: -0.0193028, -0.0155805, -0.0190944, -0.0155594, -0.0027753, 0.0024279
8: 0.9684862, 0.9791510, 0.9690833, 0.9792114, -0.0082295, 0.0073351
9: 0.0037807, 0.0069151, 0.0037629, 0.0067396, -0.0020728, 0.0023587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055401, upper bound: 0.0054090
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055401, upper bound: 0.0054672
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002978, 0.0000650, -0.0002884, 0.0001201, -0.0002951, 0.0002388
1: -0.0000738, 0.0014213, -0.0000302, 0.0015057, -0.0010800, 0.0010150
2: 0.0142114, 0.0164504, 0.0140850, 0.0163851, -0.0014962, 0.0016072
3: 0.0000595, 0.0017431, -0.0000356, 0.0016940, -0.0011151, 0.0012038
4: -0.0043248, -0.0027718, -0.0044125, -0.0028171, -0.0011150, 0.0011485
5: 0.0079976, 0.0096781, 0.0079026, 0.0096291, -0.0011121, 0.0012012
6: 0.0092807, 0.0099149, 0.0092992, 0.0099507, -0.0005785, 0.0006157
7: -0.0194096, -0.0157613, -0.0193032, -0.0155553, -0.0025596, 0.0023135
8: 0.9681801, 0.9786329, 0.9684850, 0.9792233, -0.0075136, 0.0070093
9: 0.0039329, 0.0070050, 0.0037594, 0.0069154, -0.0019777, 0.0021708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050567
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050567
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002884, 0.0001134, -0.0002888, 0.0001309, -0.0003059, 0.0002784
1: -0.0000300, 0.0014954, -0.0000316, 0.0015222, -0.0011336, 0.0010696
2: 0.0141005, 0.0163849, 0.0140604, 0.0163873, -0.0015722, 0.0016758
3: -0.0000239, 0.0016938, -0.0000541, 0.0016956, -0.0011696, 0.0012507
4: -0.0044017, -0.0028172, -0.0044295, -0.0028156, -0.0011856, 0.0012341
5: 0.0079143, 0.0096289, 0.0078842, 0.0096307, -0.0011663, 0.0012475
6: 0.0092992, 0.0099463, 0.0092986, 0.0099576, -0.0006584, 0.0006477
7: -0.0193028, -0.0155805, -0.0193067, -0.0155152, -0.0026142, 0.0024145
8: 0.9684862, 0.9791510, 0.9684749, 0.9793380, -0.0078449, 0.0073699
9: 0.0037807, 0.0069151, 0.0037257, 0.0069184, -0.0020677, 0.0022291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055407, upper bound: 0.0054099
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055407, upper bound: 0.0054672
time: 0.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.59 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0051559, upper bound: 0.0050322
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0052356, upper bound: 0.0050322
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0052622, upper bound: 0.0054135
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0052622, upper bound: 0.0054807
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0051545, upper bound: 0.0053090
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0053089
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0052622, upper bound: 0.0056821
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0052622, upper bound: 0.0057467
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050322
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050322
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0055401, upper bound: 0.0054090
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0055401, upper bound: 0.0054672
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050567
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050567
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0055407, upper bound: 0.0054099
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 8, lower bound: -0.0055407, upper bound: 0.0054672

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000534, -0.0002695, 0.0000794, -0.0002513, 0.0002203
1: 0.0000122, 0.0014035, 0.0000587, 0.0014434, -0.0009835, 0.0009276
2: 0.0142381, 0.0163217, 0.0141784, 0.0162521, -0.0013761, 0.0014658
3: 0.0000795, 0.0016463, 0.0000347, 0.0015940, -0.0010287, 0.0010990
4: -0.0043063, -0.0028611, -0.0043477, -0.0029094, -0.0009971, 0.0010410
5: 0.0080176, 0.0095815, 0.0079728, 0.0095292, -0.0010263, 0.0010967
6: 0.0093171, 0.0099073, 0.0093368, 0.0099242, -0.0005094, 0.0005386
7: -0.0191998, -0.0158047, -0.0190864, -0.0157075, -0.0023438, 0.0021744
8: 0.9687811, 0.9785085, 0.9691061, 0.9787872, -0.0068502, 0.0064380
9: 0.0039695, 0.0068284, 0.0038876, 0.0067328, -0.0018474, 0.0019859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051559, upper bound: 0.0049907
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051559, upper bound: 0.0050322
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000536, -0.0002715, 0.0000871, -0.0002593, 0.0002233
1: 0.0000123, 0.0014037, 0.0000491, 0.0014550, -0.0010007, 0.0009462
2: 0.0142378, 0.0163216, 0.0141609, 0.0162664, -0.0014042, 0.0014915
3: 0.0000793, 0.0016463, 0.0000215, 0.0016048, -0.0010502, 0.0011184
4: -0.0043065, -0.0028611, -0.0043598, -0.0028994, -0.0010175, 0.0010591
5: 0.0080173, 0.0095814, 0.0079596, 0.0095400, -0.0010477, 0.0011160
6: 0.0093172, 0.0099074, 0.0093328, 0.0099292, -0.0005170, 0.0005582
7: -0.0191997, -0.0158042, -0.0191098, -0.0156790, -0.0023856, 0.0022117
8: 0.9687815, 0.9785099, 0.9690391, 0.9788688, -0.0069704, 0.0065685
9: 0.0039691, 0.0068283, 0.0038636, 0.0067526, -0.0018827, 0.0020212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052356, upper bound: 0.0049907
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052356, upper bound: 0.0050322
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001025, -0.0002794, 0.0000558, -0.0002238, 0.0002768
1: 0.0000571, 0.0014787, 0.0000121, 0.0014072, -0.0009568, 0.0010382
2: 0.0141255, 0.0162544, 0.0142326, 0.0163219, -0.0015471, 0.0014144
3: -0.0000051, 0.0015958, 0.0000754, 0.0016465, -0.0011600, 0.0010553
4: -0.0043844, -0.0029077, -0.0043101, -0.0028609, -0.0010992, 0.0010423
5: 0.0079330, 0.0095310, 0.0080135, 0.0095817, -0.0011576, 0.0010526
6: 0.0093362, 0.0099392, 0.0093171, 0.0099088, -0.0005727, 0.0005385
7: -0.0190902, -0.0156212, -0.0192002, -0.0157958, -0.0022038, 0.0024741
8: 0.9690952, 0.9790342, 0.9687800, 0.9785340, -0.0066220, 0.0072306
9: 0.0038150, 0.0067361, 0.0039620, 0.0068287, -0.0020963, 0.0018798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0051559
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0052356
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001025, -0.0002698, 0.0001025, -0.0002606, 0.0002606
1: 0.0000571, 0.0014787, 0.0000571, 0.0014787, -0.0009902, 0.0009902
2: 0.0141255, 0.0162544, 0.0141255, 0.0162544, -0.0014627, 0.0014627
3: -0.0000051, 0.0015958, -0.0000051, 0.0015958, -0.0010912, 0.0010912
4: -0.0043844, -0.0029077, -0.0043844, -0.0029077, -0.0010821, 0.0010821
5: 0.0079330, 0.0095310, 0.0079330, 0.0095310, -0.0010884, 0.0010884
6: 0.0093362, 0.0099392, 0.0093362, 0.0099392, -0.0006030, 0.0006030
7: -0.0190902, -0.0156212, -0.0190902, -0.0156212, -0.0022826, 0.0022826
8: 0.9690952, 0.9790342, 0.9690952, 0.9790342, -0.0068496, 0.0068496
9: 0.0038150, 0.0067361, 0.0038150, 0.0067361, -0.0019461, 0.0019461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0052503
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0053374
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000534, -0.0002880, 0.0000906, -0.0002683, 0.0002444
1: 0.0000122, 0.0014035, -0.0000280, 0.0014605, -0.0010309, 0.0010375
2: 0.0142381, 0.0163217, 0.0141527, 0.0163819, -0.0015420, 0.0015367
3: 0.0000795, 0.0016463, 0.0000154, 0.0016916, -0.0011543, 0.0011523
4: -0.0043063, -0.0028611, -0.0043655, -0.0028193, -0.0011067, 0.0010902
5: 0.0080176, 0.0095815, 0.0079535, 0.0096267, -0.0011517, 0.0011499
6: 0.0093171, 0.0099073, 0.0093001, 0.0099315, -0.0005295, 0.0005832
7: -0.0191998, -0.0158047, -0.0192978, -0.0156656, -0.0024594, 0.0024492
8: 0.9687811, 0.9785085, 0.9685001, 0.9789071, -0.0071813, 0.0072109
9: 0.0039695, 0.0068284, 0.0038524, 0.0069109, -0.0020777, 0.0020832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051545, upper bound: 0.0052353
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051545, upper bound: 0.0053089
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000536, -0.0002902, 0.0000972, -0.0002758, 0.0002474
1: 0.0000123, 0.0014037, -0.0000384, 0.0014705, -0.0010478, 0.0010580
2: 0.0142378, 0.0163216, 0.0141377, 0.0163974, -0.0015719, 0.0015620
3: 0.0000793, 0.0016463, 0.0000041, 0.0017033, -0.0011765, 0.0011714
4: -0.0043065, -0.0028611, -0.0043759, -0.0028085, -0.0011304, 0.0011080
5: 0.0080173, 0.0095814, 0.0079422, 0.0096384, -0.0011738, 0.0011690
6: 0.0093172, 0.0099074, 0.0092957, 0.0099357, -0.0005369, 0.0005997
7: -0.0191997, -0.0158042, -0.0193233, -0.0156412, -0.0025006, 0.0024865
8: 0.9687815, 0.9785099, 0.9684274, 0.9789771, -0.0072997, 0.0073511
9: 0.0039691, 0.0068283, 0.0038318, 0.0069323, -0.0021132, 0.0021180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0052353
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0053089
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001025, -0.0002978, 0.0000650, -0.0002406, 0.0003050
1: 0.0000571, 0.0014787, -0.0000738, 0.0014213, -0.0010097, 0.0011657
2: 0.0141255, 0.0162544, 0.0142114, 0.0164504, -0.0017398, 0.0014937
3: -0.0000051, 0.0015958, 0.0000595, 0.0017431, -0.0013055, 0.0011150
4: -0.0043844, -0.0029077, -0.0043248, -0.0027718, -0.0012272, 0.0010973
5: 0.0079330, 0.0095310, 0.0079976, 0.0096781, -0.0013028, 0.0011121
6: 0.0093362, 0.0099392, 0.0092807, 0.0099149, -0.0005787, 0.0005829
7: -0.0190902, -0.0156212, -0.0194096, -0.0157613, -0.0023330, 0.0027952
8: 0.9690952, 0.9790342, 0.9681801, 0.9786329, -0.0069922, 0.0081285
9: 0.0038150, 0.0067361, 0.0039329, 0.0070050, -0.0023649, 0.0019886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0053957
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0054927
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001025, -0.0002884, 0.0001134, -0.0002783, 0.0002878
1: 0.0000571, 0.0014787, -0.0000300, 0.0014954, -0.0010461, 0.0011041
2: 0.0141255, 0.0162544, 0.0141005, 0.0163849, -0.0016375, 0.0015463
3: -0.0000051, 0.0015958, -0.0000239, 0.0016938, -0.0012247, 0.0011541
4: -0.0043844, -0.0029077, -0.0044017, -0.0028172, -0.0011890, 0.0011401
5: 0.0079330, 0.0095310, 0.0079143, 0.0096289, -0.0012219, 0.0011512
6: 0.0093362, 0.0099392, 0.0092992, 0.0099463, -0.0006101, 0.0006400
7: -0.0190902, -0.0156212, -0.0193028, -0.0155805, -0.0024190, 0.0025901
8: 0.9690952, 0.9790342, 0.9684862, 0.9791510, -0.0072402, 0.0076611
9: 0.0038150, 0.0067361, 0.0037807, 0.0069151, -0.0022011, 0.0020609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0054710
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0055926
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002977, 0.0000626, -0.0002695, 0.0000794, -0.0002795, 0.0002371
1: -0.0000736, 0.0014176, 0.0000587, 0.0014434, -0.0011115, 0.0009811
2: 0.0142170, 0.0164502, 0.0141784, 0.0162521, -0.0014562, 0.0016589
3: 0.0000637, 0.0017429, 0.0000347, 0.0015940, -0.0010890, 0.0012448
4: -0.0043209, -0.0027719, -0.0043477, -0.0029094, -0.0010527, 0.0011697
5: 0.0080017, 0.0096779, 0.0079728, 0.0095292, -0.0010865, 0.0012423
6: 0.0092807, 0.0099133, 0.0093368, 0.0099242, -0.0005545, 0.0005613
7: -0.0194092, -0.0157703, -0.0190864, -0.0157075, -0.0026652, 0.0023049
8: 0.9681813, 0.9786072, 0.9691061, 0.9787872, -0.0077504, 0.0068119
9: 0.0039405, 0.0070047, 0.0038876, 0.0067328, -0.0019573, 0.0022549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0049907
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050322
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002977, 0.0000628, -0.0002715, 0.0000871, -0.0002876, 0.0002399
1: -0.0000735, 0.0014179, 0.0000491, 0.0014550, -0.0011287, 0.0009984
2: 0.0142165, 0.0164501, 0.0141609, 0.0162664, -0.0014824, 0.0016845
3: 0.0000633, 0.0017429, 0.0000215, 0.0016048, -0.0011090, 0.0012640
4: -0.0043212, -0.0027720, -0.0043598, -0.0028994, -0.0010718, 0.0011876
5: 0.0080014, 0.0096779, 0.0079596, 0.0095400, -0.0011065, 0.0012615
6: 0.0092808, 0.0099134, 0.0093328, 0.0099292, -0.0005622, 0.0005803
7: -0.0194090, -0.0157696, -0.0191098, -0.0156790, -0.0027069, 0.0023392
8: 0.9681816, 0.9786091, 0.9690391, 0.9788688, -0.0078701, 0.0069338
9: 0.0039399, 0.0070046, 0.0038636, 0.0067526, -0.0019900, 0.0022900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0049907
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050322
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002884, 0.0001134, -0.0002794, 0.0000558, -0.0002477, 0.0002924
1: -0.0000300, 0.0014954, 0.0000121, 0.0014072, -0.0010627, 0.0010802
2: 0.0141005, 0.0163849, 0.0142326, 0.0163219, -0.0016100, 0.0015755
3: -0.0000239, 0.0016938, 0.0000754, 0.0016465, -0.0012073, 0.0011773
4: -0.0044017, -0.0028172, -0.0043101, -0.0028609, -0.0011428, 0.0011452
5: 0.0079143, 0.0096289, 0.0080135, 0.0095817, -0.0012048, 0.0011744
6: 0.0092992, 0.0099463, 0.0093171, 0.0099088, -0.0006096, 0.0005563
7: -0.0193028, -0.0155805, -0.0192002, -0.0157958, -0.0024760, 0.0025765
8: 0.9684862, 0.9791510, 0.9687800, 0.9785340, -0.0073719, 0.0075241
9: 0.0037807, 0.0069151, 0.0039620, 0.0068287, -0.0021825, 0.0021066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053090, upper bound: 0.0051545
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053089, upper bound: 0.0052316
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002884, 0.0001134, -0.0002698, 0.0001025, -0.0002878, 0.0002783
1: -0.0000300, 0.0014954, 0.0000571, 0.0014787, -0.0011041, 0.0010461
2: 0.0141005, 0.0163849, 0.0141255, 0.0162544, -0.0015463, 0.0016375
3: -0.0000239, 0.0016938, -0.0000051, 0.0015958, -0.0011541, 0.0012247
4: -0.0044017, -0.0028172, -0.0043844, -0.0029077, -0.0011401, 0.0011890
5: 0.0079143, 0.0096289, 0.0079330, 0.0095310, -0.0011512, 0.0012219
6: 0.0092992, 0.0099463, 0.0093362, 0.0099392, -0.0006400, 0.0006101
7: -0.0193028, -0.0155805, -0.0190902, -0.0156212, -0.0025901, 0.0024190
8: 0.9684862, 0.9791510, 0.9690952, 0.9790342, -0.0076611, 0.0072402
9: 0.0037807, 0.0069151, 0.0038150, 0.0067361, -0.0020609, 0.0022011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053090, upper bound: 0.0052456
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053089, upper bound: 0.0053259
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002977, 0.0000626, -0.0002880, 0.0000906, -0.0002672, 0.0002351
1: -0.0000736, 0.0014176, -0.0000280, 0.0014605, -0.0010351, 0.0009791
2: 0.0142170, 0.0164502, 0.0141527, 0.0163819, -0.0014505, 0.0015404
3: 0.0000637, 0.0017429, 0.0000154, 0.0016916, -0.0010837, 0.0011539
4: -0.0043209, -0.0027719, -0.0043655, -0.0028193, -0.0010577, 0.0011004
5: 0.0080017, 0.0096779, 0.0079535, 0.0096267, -0.0010811, 0.0011514
6: 0.0092807, 0.0099133, 0.0093001, 0.0099315, -0.0005534, 0.0005859
7: -0.0194092, -0.0157703, -0.0192978, -0.0156656, -0.0024538, 0.0022817
8: 0.9681813, 0.9786072, 0.9685001, 0.9789071, -0.0072014, 0.0067878
9: 0.0039405, 0.0070047, 0.0038524, 0.0069109, -0.0019409, 0.0020810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050093
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050567
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002977, 0.0000628, -0.0002902, 0.0000972, -0.0002741, 0.0002380
1: -0.0000735, 0.0014179, -0.0000384, 0.0014705, -0.0010496, 0.0009959
2: 0.0142165, 0.0164501, 0.0141377, 0.0163974, -0.0014764, 0.0015620
3: 0.0000633, 0.0017429, 0.0000041, 0.0017033, -0.0011036, 0.0011701
4: -0.0043212, -0.0027720, -0.0043759, -0.0028085, -0.0010758, 0.0011156
5: 0.0080014, 0.0096779, 0.0079422, 0.0096384, -0.0011009, 0.0011675
6: 0.0092808, 0.0099134, 0.0092957, 0.0099357, -0.0005600, 0.0006028
7: -0.0194090, -0.0157696, -0.0193233, -0.0156412, -0.0024887, 0.0023178
8: 0.9681816, 0.9786091, 0.9684274, 0.9789771, -0.0073025, 0.0069077
9: 0.0039399, 0.0070046, 0.0038318, 0.0069323, -0.0019747, 0.0021104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050093
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050567
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002884, 0.0001134, -0.0002978, 0.0000650, -0.0002387, 0.0002932
1: -0.0000300, 0.0014954, -0.0000738, 0.0014213, -0.0010095, 0.0010905
2: 0.0141005, 0.0163849, 0.0142114, 0.0164504, -0.0016228, 0.0014898
3: -0.0000239, 0.0016938, 0.0000595, 0.0017431, -0.0012156, 0.0011109
4: -0.0044017, -0.0028172, -0.0043248, -0.0027718, -0.0011594, 0.0011052
5: 0.0079143, 0.0096289, 0.0079976, 0.0096781, -0.0012130, 0.0011080
6: 0.0092992, 0.0099463, 0.0092807, 0.0099149, -0.0006156, 0.0005829
7: -0.0193028, -0.0155805, -0.0194096, -0.0157613, -0.0023112, 0.0025851
8: 0.9684862, 0.9791510, 0.9681801, 0.9786329, -0.0069769, 0.0075867
9: 0.0037807, 0.0069151, 0.0039329, 0.0070050, -0.0021923, 0.0019740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053106, upper bound: 0.0051567
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053106, upper bound: 0.0052319
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002884, 0.0001134, -0.0002884, 0.0001134, -0.0002780, 0.0002780
1: -0.0000300, 0.0014954, -0.0000300, 0.0014954, -0.0010516, 0.0010516
2: 0.0141005, 0.0163849, 0.0141005, 0.0163849, -0.0015509, 0.0015509
3: -0.0000239, 0.0016938, -0.0000239, 0.0016938, -0.0011560, 0.0011560
4: -0.0044017, -0.0028172, -0.0044017, -0.0028172, -0.0011551, 0.0011551
5: 0.0079143, 0.0096289, 0.0079143, 0.0096289, -0.0011529, 0.0011529
6: 0.0092992, 0.0099463, 0.0092992, 0.0099463, -0.0006471, 0.0006471
7: -0.0193028, -0.0155805, -0.0193028, -0.0155805, -0.0024055, 0.0024055
8: 0.9684862, 0.9791510, 0.9684862, 0.9791510, -0.0072647, 0.0072647
9: 0.0037807, 0.0069151, 0.0037807, 0.0069151, -0.0020545, 0.0020545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053106, upper bound: 0.0052457
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053106, upper bound: 0.0053259
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.69 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0051559, upper bound: 0.0049907
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0051559, upper bound: 0.0050322
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0052356, upper bound: 0.0049907
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0052356, upper bound: 0.0050322
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0051559
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0052356
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0052503
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0053374
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0051545, upper bound: 0.0052353
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0051545, upper bound: 0.0053089
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0052353
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0053089
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0053957
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0054927
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0054710
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0050322, upper bound: 0.0055926
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0049907
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050322
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0049907
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050322
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053090, upper bound: 0.0051545
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053089, upper bound: 0.0052316
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053090, upper bound: 0.0052456
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053089, upper bound: 0.0053259
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050093
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053957, upper bound: 0.0050567
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050093
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0054927, upper bound: 0.0050567
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053106, upper bound: 0.0051567
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053106, upper bound: 0.0052319
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053106, upper bound: 0.0052457
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 8, lower bound: -0.0053106, upper bound: 0.0053259

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002695, 0.0000794, -0.0002503, 0.0001950
1: 0.0000141, 0.0013627, 0.0000587, 0.0014434, -0.0009633, 0.0008838
2: 0.0142991, 0.0163189, 0.0141784, 0.0162521, -0.0013105, 0.0014395
3: 0.0001254, 0.0016442, 0.0000347, 0.0015940, -0.0009794, 0.0010811
4: -0.0042639, -0.0028630, -0.0043477, -0.0029094, -0.0009516, 0.0010079
5: 0.0080634, 0.0095794, 0.0079728, 0.0095292, -0.0009771, 0.0010790
6: 0.0093179, 0.0098900, 0.0093368, 0.0099242, -0.0004405, 0.0005200
7: -0.0191953, -0.0159042, -0.0190864, -0.0157075, -0.0023261, 0.0020674
8: 0.9687942, 0.9782236, 0.9691061, 0.9787872, -0.0067237, 0.0061317
9: 0.0040533, 0.0068246, 0.0038876, 0.0067328, -0.0017573, 0.0019641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050005
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050005
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002695, 0.0000794, -0.0002547, 0.0002036
1: 0.0000058, 0.0013729, 0.0000587, 0.0014434, -0.0009844, 0.0009056
2: 0.0142839, 0.0163313, 0.0141784, 0.0162521, -0.0013430, 0.0014709
3: 0.0001140, 0.0016536, 0.0000347, 0.0015940, -0.0010039, 0.0011047
4: -0.0042745, -0.0028544, -0.0043477, -0.0029094, -0.0009742, 0.0010319
5: 0.0080519, 0.0095887, 0.0079728, 0.0095292, -0.0010015, 0.0011025
6: 0.0093144, 0.0098943, 0.0093368, 0.0099242, -0.0004593, 0.0005292
7: -0.0192155, -0.0158793, -0.0190864, -0.0157075, -0.0023771, 0.0021205
8: 0.9687363, 0.9782947, 0.9691061, 0.9787872, -0.0068701, 0.0062836
9: 0.0040323, 0.0068416, 0.0038876, 0.0067328, -0.0018020, 0.0020070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050322
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050322
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002715, 0.0000871, -0.0002579, 0.0001976
1: 0.0000141, 0.0013627, 0.0000491, 0.0014550, -0.0009767, 0.0009010
2: 0.0142991, 0.0163189, 0.0141609, 0.0162664, -0.0013365, 0.0014595
3: 0.0001254, 0.0016442, 0.0000215, 0.0016048, -0.0009993, 0.0010961
4: -0.0042639, -0.0028630, -0.0043598, -0.0028994, -0.0009706, 0.0010217
5: 0.0080634, 0.0095794, 0.0079596, 0.0095400, -0.0009970, 0.0010940
6: 0.0093179, 0.0098900, 0.0093328, 0.0099292, -0.0004461, 0.0005390
7: -0.0191953, -0.0159042, -0.0191098, -0.0156790, -0.0023587, 0.0021015
8: 0.9687942, 0.9782236, 0.9690391, 0.9788688, -0.0068170, 0.0062527
9: 0.0040533, 0.0068246, 0.0038636, 0.0067526, -0.0017899, 0.0019916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0049907
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0049907
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002715, 0.0000871, -0.0002582, 0.0002027
1: 0.0000058, 0.0013729, 0.0000491, 0.0014550, -0.0009839, 0.0009077
2: 0.0142839, 0.0163313, 0.0141609, 0.0162664, -0.0013446, 0.0014695
3: 0.0001140, 0.0016536, 0.0000215, 0.0016048, -0.0010049, 0.0011031
4: -0.0042745, -0.0028544, -0.0043598, -0.0028994, -0.0009799, 0.0010321
5: 0.0080519, 0.0095887, 0.0079596, 0.0095400, -0.0010024, 0.0011010
6: 0.0093144, 0.0098943, 0.0093328, 0.0099292, -0.0004626, 0.0005483
7: -0.0192155, -0.0158793, -0.0191098, -0.0156790, -0.0023683, 0.0021123
8: 0.9687363, 0.9782947, 0.9690391, 0.9788688, -0.0068645, 0.0062925
9: 0.0040323, 0.0068416, 0.0038636, 0.0067526, -0.0017978, 0.0020014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050322
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050322
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002794, 0.0000534, -0.0002202, 0.0002499
1: 0.0000590, 0.0014349, 0.0000122, 0.0014035, -0.0009235, 0.0009947
2: 0.0141911, 0.0162516, 0.0142381, 0.0163217, -0.0014825, 0.0013708
3: 0.0000442, 0.0015936, 0.0000795, 0.0016463, -0.0011116, 0.0010256
4: -0.0043389, -0.0029096, -0.0043063, -0.0028611, -0.0010526, 0.0009899
5: 0.0079823, 0.0095289, 0.0080176, 0.0095815, -0.0011093, 0.0010233
6: 0.0093370, 0.0099206, 0.0093171, 0.0099073, -0.0005225, 0.0005141
7: -0.0190857, -0.0157282, -0.0191998, -0.0158047, -0.0021721, 0.0023712
8: 0.9691080, 0.9787278, 0.9687811, 0.9785085, -0.0064125, 0.0069286
9: 0.0039050, 0.0067323, 0.0039695, 0.0068284, -0.0020089, 0.0018442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0051559
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0051559
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002794, 0.0000536, -0.0002232, 0.0002573
1: 0.0000494, 0.0014451, 0.0000123, 0.0014037, -0.0009418, 0.0010119
2: 0.0141759, 0.0162660, 0.0142378, 0.0163216, -0.0015082, 0.0013988
3: 0.0000328, 0.0016044, 0.0000793, 0.0016463, -0.0011310, 0.0010467
4: -0.0043494, -0.0028997, -0.0043065, -0.0028611, -0.0010707, 0.0010094
5: 0.0079709, 0.0095397, 0.0080173, 0.0095814, -0.0011286, 0.0010443
6: 0.0093329, 0.0099249, 0.0093172, 0.0099074, -0.0005395, 0.0005217
7: -0.0191090, -0.0157033, -0.0191997, -0.0158042, -0.0022090, 0.0024129
8: 0.9690413, 0.9787990, 0.9687815, 0.9785099, -0.0065423, 0.0070485
9: 0.0038841, 0.0067519, 0.0039691, 0.0068283, -0.0020442, 0.0018789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052356
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052356
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002698, 0.0001001, -0.0002571, 0.0002329
1: 0.0000590, 0.0014349, 0.0000573, 0.0014751, -0.0009549, 0.0009429
2: 0.0141911, 0.0162516, 0.0141309, 0.0162542, -0.0013925, 0.0014186
3: 0.0000442, 0.0015936, -0.0000010, 0.0015956, -0.0010388, 0.0010621
4: -0.0043389, -0.0029096, -0.0043806, -0.0029079, -0.0010311, 0.0010248
5: 0.0079823, 0.0095289, 0.0079371, 0.0095308, -0.0010361, 0.0010597
6: 0.0093370, 0.0099206, 0.0093362, 0.0099377, -0.0005486, 0.0005844
7: -0.0190857, -0.0157282, -0.0190899, -0.0156301, -0.0022521, 0.0021721
8: 0.9691080, 0.9787278, 0.9690962, 0.9790087, -0.0066338, 0.0065211
9: 0.0039050, 0.0067323, 0.0038225, 0.0067358, -0.0018523, 0.0019118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052134, upper bound: 0.0052503
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052134, upper bound: 0.0052503
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002698, 0.0001003, -0.0002613, 0.0002407
1: 0.0000494, 0.0014451, 0.0000573, 0.0014753, -0.0009790, 0.0009610
2: 0.0141759, 0.0162660, 0.0141305, 0.0162541, -0.0014195, 0.0014542
3: 0.0000328, 0.0016044, -0.0000013, 0.0015955, -0.0010591, 0.0010882
4: -0.0043494, -0.0028997, -0.0043809, -0.0029079, -0.0010498, 0.0010496
5: 0.0079709, 0.0095397, 0.0079368, 0.0095308, -0.0010563, 0.0010857
6: 0.0093329, 0.0099249, 0.0093363, 0.0099378, -0.0005660, 0.0005887
7: -0.0191090, -0.0157033, -0.0190897, -0.0156295, -0.0022997, 0.0022155
8: 0.9690413, 0.9787990, 0.9690965, 0.9790106, -0.0068012, 0.0066472
9: 0.0038841, 0.0067519, 0.0038219, 0.0067357, -0.0018889, 0.0019547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052134, upper bound: 0.0053374
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052134, upper bound: 0.0053374
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002880, 0.0000906, -0.0002673, 0.0002190
1: 0.0000141, 0.0013627, -0.0000280, 0.0014605, -0.0010106, 0.0009936
2: 0.0142991, 0.0163189, 0.0141527, 0.0163819, -0.0014764, 0.0015104
3: 0.0001254, 0.0016442, 0.0000154, 0.0016916, -0.0011050, 0.0011344
4: -0.0042639, -0.0028630, -0.0043655, -0.0028193, -0.0010612, 0.0010571
5: 0.0080634, 0.0095794, 0.0079535, 0.0096267, -0.0011025, 0.0011322
6: 0.0093179, 0.0098900, 0.0093001, 0.0099315, -0.0004606, 0.0005646
7: -0.0191953, -0.0159042, -0.0192978, -0.0156656, -0.0024417, 0.0023423
8: 0.9687942, 0.9782236, 0.9685001, 0.9789071, -0.0070547, 0.0069046
9: 0.0040533, 0.0068246, 0.0038524, 0.0069109, -0.0019877, 0.0020614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052791
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052791
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002880, 0.0000906, -0.0002717, 0.0002277
1: 0.0000058, 0.0013729, -0.0000280, 0.0014605, -0.0010317, 0.0010154
2: 0.0142839, 0.0163313, 0.0141527, 0.0163819, -0.0015090, 0.0015418
3: 0.0001140, 0.0016536, 0.0000154, 0.0016916, -0.0011294, 0.0011580
4: -0.0042745, -0.0028544, -0.0043655, -0.0028193, -0.0010838, 0.0010811
5: 0.0080519, 0.0095887, 0.0079535, 0.0096267, -0.0011269, 0.0011557
6: 0.0093144, 0.0098943, 0.0093001, 0.0099315, -0.0004794, 0.0005738
7: -0.0192155, -0.0158793, -0.0192978, -0.0156656, -0.0024926, 0.0023953
8: 0.9687363, 0.9782947, 0.9685001, 0.9789071, -0.0072011, 0.0070565
9: 0.0040323, 0.0068416, 0.0038524, 0.0069109, -0.0020324, 0.0021043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053090
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053090
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002902, 0.0000972, -0.0002734, 0.0002217
1: 0.0000141, 0.0013627, -0.0000384, 0.0014705, -0.0010220, 0.0010128
2: 0.0142991, 0.0163189, 0.0141377, 0.0163974, -0.0015042, 0.0015275
3: 0.0001254, 0.0016442, 0.0000041, 0.0017033, -0.0011256, 0.0011472
4: -0.0042639, -0.0028630, -0.0043759, -0.0028085, -0.0010835, 0.0010689
5: 0.0080634, 0.0095794, 0.0079422, 0.0096384, -0.0011230, 0.0011450
6: 0.0093179, 0.0098900, 0.0092957, 0.0099357, -0.0004654, 0.0005806
7: -0.0191953, -0.0159042, -0.0193233, -0.0156412, -0.0024694, 0.0023763
8: 0.9687942, 0.9782236, 0.9684274, 0.9789771, -0.0071341, 0.0070353
9: 0.0040533, 0.0068246, 0.0038318, 0.0069323, -0.0020204, 0.0020848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052352
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052352
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002902, 0.0000972, -0.0002746, 0.0002264
1: 0.0000058, 0.0013729, -0.0000384, 0.0014705, -0.0010310, 0.0010141
2: 0.0142839, 0.0163313, 0.0141377, 0.0163974, -0.0015057, 0.0015400
3: 0.0001140, 0.0016536, 0.0000041, 0.0017033, -0.0011264, 0.0011562
4: -0.0042745, -0.0028544, -0.0043759, -0.0028085, -0.0010848, 0.0010810
5: 0.0080519, 0.0095887, 0.0079422, 0.0096384, -0.0011238, 0.0011539
6: 0.0093144, 0.0098943, 0.0092957, 0.0099357, -0.0004826, 0.0005886
7: -0.0192155, -0.0158793, -0.0193233, -0.0156412, -0.0024832, 0.0023820
8: 0.9687363, 0.9782947, 0.9684274, 0.9789771, -0.0071938, 0.0070426
9: 0.0040323, 0.0068416, 0.0038318, 0.0069323, -0.0020229, 0.0020982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053089
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053089
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002977, 0.0000626, -0.0002370, 0.0002782
1: 0.0000590, 0.0014349, -0.0000736, 0.0014176, -0.0009770, 0.0011227
2: 0.0141911, 0.0162516, 0.0142170, 0.0164502, -0.0016756, 0.0014509
3: 0.0000442, 0.0015936, 0.0000637, 0.0017429, -0.0012574, 0.0010859
4: -0.0043389, -0.0029096, -0.0043209, -0.0027719, -0.0011814, 0.0010454
5: 0.0079823, 0.0095289, 0.0080017, 0.0096779, -0.0012549, 0.0010835
6: 0.0093370, 0.0099206, 0.0092807, 0.0099133, -0.0005452, 0.0005592
7: -0.0190857, -0.0157282, -0.0194092, -0.0157703, -0.0023026, 0.0026925
8: 0.9691080, 0.9787278, 0.9681813, 0.9786072, -0.0067864, 0.0078288
9: 0.0039050, 0.0067323, 0.0039405, 0.0070047, -0.0022779, 0.0019541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053957
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053957
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002977, 0.0000628, -0.0002398, 0.0002855
1: 0.0000494, 0.0014451, -0.0000735, 0.0014179, -0.0009940, 0.0011398
2: 0.0141759, 0.0162660, 0.0142165, 0.0164501, -0.0017012, 0.0014770
3: 0.0000328, 0.0016044, 0.0000633, 0.0017429, -0.0012766, 0.0011055
4: -0.0043494, -0.0028997, -0.0043212, -0.0027720, -0.0011992, 0.0010637
5: 0.0079709, 0.0095397, 0.0080014, 0.0096779, -0.0012740, 0.0011030
6: 0.0093329, 0.0099249, 0.0092808, 0.0099134, -0.0005617, 0.0005670
7: -0.0191090, -0.0157033, -0.0194090, -0.0157696, -0.0023365, 0.0027341
8: 0.9690413, 0.9787990, 0.9681816, 0.9786091, -0.0069076, 0.0079482
9: 0.0038841, 0.0067519, 0.0039399, 0.0070046, -0.0023130, 0.0019862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0054927
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0054927
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002884, 0.0001110, -0.0002748, 0.0002601
1: 0.0000590, 0.0014349, -0.0000298, 0.0014918, -0.0010110, 0.0010574
2: 0.0141911, 0.0162516, 0.0141059, 0.0163846, -0.0015681, 0.0015026
3: 0.0000442, 0.0015936, -0.0000198, 0.0016936, -0.0011728, 0.0011252
4: -0.0043389, -0.0029096, -0.0043979, -0.0028174, -0.0011391, 0.0010831
5: 0.0079823, 0.0095289, 0.0079184, 0.0096287, -0.0011701, 0.0011227
6: 0.0093370, 0.0099206, 0.0092993, 0.0099447, -0.0005724, 0.0006213
7: -0.0190857, -0.0157282, -0.0193023, -0.0155894, -0.0023889, 0.0024798
8: 0.9691080, 0.9787278, 0.9684874, 0.9791255, -0.0070258, 0.0073363
9: 0.0039050, 0.0067323, 0.0037882, 0.0069147, -0.0021076, 0.0020270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052008, upper bound: 0.0054710
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052008, upper bound: 0.0054710
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002884, 0.0001111, -0.0002790, 0.0002679
1: 0.0000494, 0.0014451, -0.0000297, 0.0014920, -0.0010344, 0.0010752
2: 0.0141759, 0.0162660, 0.0141056, 0.0163845, -0.0015947, 0.0015373
3: 0.0000328, 0.0016044, -0.0000201, 0.0016936, -0.0011928, 0.0011506
4: -0.0043494, -0.0028997, -0.0043981, -0.0028175, -0.0011576, 0.0011072
5: 0.0079709, 0.0095397, 0.0079181, 0.0096287, -0.0011900, 0.0011480
6: 0.0093329, 0.0099249, 0.0092993, 0.0099448, -0.0005895, 0.0006256
7: -0.0191090, -0.0157033, -0.0193022, -0.0155889, -0.0024350, 0.0025229
8: 0.9690413, 0.9787990, 0.9684879, 0.9791269, -0.0071889, 0.0074608
9: 0.0038841, 0.0067519, 0.0037878, 0.0069146, -0.0021439, 0.0020686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052008, upper bound: 0.0055926
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052008, upper bound: 0.0055926
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002695, 0.0000794, -0.0002785, 0.0002120
1: -0.0000716, 0.0013756, 0.0000587, 0.0014434, -0.0010954, 0.0009406
2: 0.0142798, 0.0164472, 0.0141784, 0.0162521, -0.0013954, 0.0016376
3: 0.0001109, 0.0017407, 0.0000347, 0.0015940, -0.0010433, 0.0012300
4: -0.0042773, -0.0027740, -0.0043477, -0.0029094, -0.0010105, 0.0011446
5: 0.0080489, 0.0096757, 0.0079728, 0.0095292, -0.0010409, 0.0012276
6: 0.0092816, 0.0098955, 0.0093368, 0.0099242, -0.0004955, 0.0005441
7: -0.0194043, -0.0158728, -0.0190864, -0.0157075, -0.0026489, 0.0022059
8: 0.9681953, 0.9783136, 0.9691061, 0.9787872, -0.0076480, 0.0065283
9: 0.0040268, 0.0070006, 0.0038876, 0.0067328, -0.0018739, 0.0022360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050005
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050005
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002695, 0.0000794, -0.0002806, 0.0002190
1: -0.0000809, 0.0013873, 0.0000587, 0.0014434, -0.0011094, 0.0009520
2: 0.0142624, 0.0164611, 0.0141784, 0.0162521, -0.0014125, 0.0016570
3: 0.0000978, 0.0017511, 0.0000347, 0.0015940, -0.0010561, 0.0012440
4: -0.0042894, -0.0027644, -0.0043477, -0.0029094, -0.0010224, 0.0011631
5: 0.0080358, 0.0096861, 0.0079728, 0.0095292, -0.0010537, 0.0012416
6: 0.0092776, 0.0099004, 0.0093368, 0.0099242, -0.0005138, 0.0005489
7: -0.0194269, -0.0158444, -0.0190864, -0.0157075, -0.0026735, 0.0022337
8: 0.9681305, 0.9783950, 0.9691061, 0.9787872, -0.0077403, 0.0066080
9: 0.0040029, 0.0070196, 0.0038876, 0.0067328, -0.0018973, 0.0022581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050322
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050322
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002715, 0.0000871, -0.0002861, 0.0002147
1: -0.0000716, 0.0013756, 0.0000491, 0.0014550, -0.0011087, 0.0009577
2: 0.0142798, 0.0164472, 0.0141609, 0.0162664, -0.0014215, 0.0016576
3: 0.0001109, 0.0017407, 0.0000215, 0.0016048, -0.0010632, 0.0012451
4: -0.0042773, -0.0027740, -0.0043598, -0.0028994, -0.0010296, 0.0011585
5: 0.0080489, 0.0096757, 0.0079596, 0.0095400, -0.0010608, 0.0012427
6: 0.0092816, 0.0098955, 0.0093328, 0.0099292, -0.0005012, 0.0005627
7: -0.0194043, -0.0158728, -0.0191098, -0.0156790, -0.0026815, 0.0022399
8: 0.9681953, 0.9783136, 0.9690391, 0.9788688, -0.0077414, 0.0066494
9: 0.0040268, 0.0070006, 0.0038636, 0.0067526, -0.0019064, 0.0022635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0049907
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0049907
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002715, 0.0000871, -0.0002864, 0.0002200
1: -0.0000809, 0.0013873, 0.0000491, 0.0014550, -0.0011149, 0.0009642
2: 0.0142624, 0.0164611, 0.0141609, 0.0162664, -0.0014292, 0.0016660
3: 0.0000978, 0.0017511, 0.0000215, 0.0016048, -0.0010684, 0.0012511
4: -0.0042894, -0.0027644, -0.0043598, -0.0028994, -0.0010385, 0.0011668
5: 0.0080358, 0.0096861, 0.0079596, 0.0095400, -0.0010659, 0.0012486
6: 0.0092776, 0.0099004, 0.0093328, 0.0099292, -0.0005170, 0.0005676
7: -0.0194269, -0.0158444, -0.0191098, -0.0156790, -0.0026905, 0.0022501
8: 0.9681305, 0.9783950, 0.9690391, 0.9788688, -0.0077816, 0.0066872
9: 0.0040029, 0.0070196, 0.0038636, 0.0067526, -0.0019138, 0.0022724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050322
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050322
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002794, 0.0000534, -0.0002443, 0.0002648
1: -0.0000279, 0.0014503, 0.0000122, 0.0014035, -0.0010338, 0.0010338
2: 0.0141680, 0.0163817, 0.0142381, 0.0163217, -0.0015411, 0.0015377
3: 0.0000268, 0.0016914, 0.0000795, 0.0016463, -0.0011557, 0.0011515
4: -0.0043549, -0.0028195, -0.0043063, -0.0028611, -0.0010933, 0.0011002
5: 0.0079650, 0.0096265, 0.0080176, 0.0095815, -0.0011533, 0.0011490
6: 0.0093001, 0.0099272, 0.0093171, 0.0099073, -0.0005650, 0.0005307
7: -0.0192976, -0.0156905, -0.0191998, -0.0158047, -0.0024476, 0.0024666
8: 0.9685011, 0.9788356, 0.9687811, 0.9785085, -0.0071897, 0.0072021
9: 0.0038734, 0.0069107, 0.0039695, 0.0068284, -0.0020893, 0.0020754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0051545
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0051545
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002794, 0.0000536, -0.0002474, 0.0002722
1: -0.0000382, 0.0014603, 0.0000123, 0.0014037, -0.0010543, 0.0010508
2: 0.0141530, 0.0163971, 0.0142378, 0.0163216, -0.0015665, 0.0015674
3: 0.0000156, 0.0017031, 0.0000793, 0.0016463, -0.0011747, 0.0011736
4: -0.0043653, -0.0028087, -0.0043065, -0.0028611, -0.0011111, 0.0011240
5: 0.0079537, 0.0096381, 0.0080173, 0.0095814, -0.0011723, 0.0011709
6: 0.0092958, 0.0099314, 0.0093172, 0.0099074, -0.0005816, 0.0005382
7: -0.0193227, -0.0156661, -0.0191997, -0.0158042, -0.0024846, 0.0025078
8: 0.9684289, 0.9789057, 0.9687815, 0.9785099, -0.0073292, 0.0073203
9: 0.0038528, 0.0069319, 0.0039691, 0.0068283, -0.0021240, 0.0021103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0052316
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0052316
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002698, 0.0001001, -0.0002844, 0.0002505
1: -0.0000279, 0.0014503, 0.0000573, 0.0014751, -0.0010757, 0.0009983
2: 0.0141680, 0.0163817, 0.0141309, 0.0162542, -0.0014754, 0.0016023
3: 0.0000268, 0.0016914, -0.0000010, 0.0015956, -0.0011012, 0.0012012
4: -0.0043549, -0.0028195, -0.0043806, -0.0029079, -0.0010886, 0.0011424
5: 0.0079650, 0.0096265, 0.0079371, 0.0095308, -0.0010984, 0.0011987
6: 0.0093001, 0.0099272, 0.0093362, 0.0099377, -0.0005900, 0.0005909
7: -0.0192976, -0.0156905, -0.0190899, -0.0156301, -0.0025618, 0.0023073
8: 0.9685011, 0.9788356, 0.9690962, 0.9790087, -0.0074889, 0.0069084
9: 0.0038734, 0.0069107, 0.0038225, 0.0067358, -0.0019661, 0.0021705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0052456
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0052456
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002698, 0.0001003, -0.0002869, 0.0002582
1: -0.0000382, 0.0014603, 0.0000573, 0.0014753, -0.0010975, 0.0010162
2: 0.0141530, 0.0163971, 0.0141305, 0.0162541, -0.0015021, 0.0016321
3: 0.0000156, 0.0017031, -0.0000013, 0.0015955, -0.0011212, 0.0012222
4: -0.0043653, -0.0028087, -0.0043809, -0.0029079, -0.0011071, 0.0011684
5: 0.0079537, 0.0096381, 0.0079368, 0.0095308, -0.0011184, 0.0012195
6: 0.0092958, 0.0099314, 0.0093363, 0.0099378, -0.0006067, 0.0005951
7: -0.0193227, -0.0156661, -0.0190897, -0.0156295, -0.0025923, 0.0023502
8: 0.9684289, 0.9789057, 0.9690965, 0.9790106, -0.0076312, 0.0070330
9: 0.0038528, 0.0069319, 0.0038219, 0.0067357, -0.0020023, 0.0022008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0053259
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0053259
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002880, 0.0000906, -0.0002662, 0.0002091
1: -0.0000716, 0.0013756, -0.0000280, 0.0014605, -0.0010123, 0.0009356
2: 0.0142798, 0.0164472, 0.0141527, 0.0163819, -0.0013855, 0.0015117
3: 0.0001109, 0.0017407, 0.0000154, 0.0016916, -0.0010348, 0.0011347
4: -0.0042773, -0.0027740, -0.0043655, -0.0028193, -0.0010126, 0.0010624
5: 0.0080489, 0.0096757, 0.0079535, 0.0096267, -0.0010323, 0.0011325
6: 0.0092816, 0.0098955, 0.0093001, 0.0099315, -0.0004790, 0.0005675
7: -0.0194043, -0.0158728, -0.0192978, -0.0156656, -0.0024354, 0.0021757
8: 0.9681953, 0.9783136, 0.9685001, 0.9789071, -0.0070616, 0.0064841
9: 0.0040268, 0.0070006, 0.0038524, 0.0069109, -0.0018516, 0.0020584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050245
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050245
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002880, 0.0000906, -0.0002705, 0.0002181
1: -0.0000809, 0.0013873, -0.0000280, 0.0014605, -0.0010333, 0.0009565
2: 0.0142624, 0.0164611, 0.0141527, 0.0163819, -0.0014167, 0.0015423
3: 0.0000978, 0.0017511, 0.0000154, 0.0016916, -0.0010583, 0.0011574
4: -0.0042894, -0.0027644, -0.0043655, -0.0028193, -0.0010343, 0.0010877
5: 0.0080358, 0.0096861, 0.0079535, 0.0096267, -0.0010557, 0.0011551
6: 0.0092776, 0.0099004, 0.0093001, 0.0099315, -0.0004996, 0.0005764
7: -0.0194269, -0.0158444, -0.0192978, -0.0156656, -0.0024848, 0.0022267
8: 0.9681305, 0.9783950, 0.9685001, 0.9789071, -0.0072058, 0.0066301
9: 0.0040029, 0.0070196, 0.0038524, 0.0069109, -0.0018946, 0.0020998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050567
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050567
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002902, 0.0000972, -0.0002731, 0.0002116
1: -0.0000716, 0.0013756, -0.0000384, 0.0014705, -0.0010257, 0.0009511
2: 0.0142798, 0.0164472, 0.0141377, 0.0163974, -0.0014094, 0.0015317
3: 0.0001109, 0.0017407, 0.0000041, 0.0017033, -0.0010532, 0.0011498
4: -0.0042773, -0.0027740, -0.0043759, -0.0028085, -0.0010293, 0.0010763
5: 0.0080489, 0.0096757, 0.0079422, 0.0096384, -0.0010506, 0.0011476
6: 0.0092816, 0.0098955, 0.0092957, 0.0099357, -0.0004847, 0.0005838
7: -0.0194043, -0.0158728, -0.0193233, -0.0156412, -0.0024681, 0.0022086
8: 0.9681953, 0.9783136, 0.9684274, 0.9789771, -0.0071554, 0.0065946
9: 0.0040268, 0.0070006, 0.0038318, 0.0069323, -0.0018827, 0.0020860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050093
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050093
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002902, 0.0000972, -0.0002730, 0.0002167
1: -0.0000809, 0.0013873, -0.0000384, 0.0014705, -0.0010310, 0.0009562
2: 0.0142624, 0.0164611, 0.0141377, 0.0163974, -0.0014151, 0.0015383
3: 0.0000978, 0.0017511, 0.0000041, 0.0017033, -0.0010567, 0.0011542
4: -0.0042894, -0.0027644, -0.0043759, -0.0028085, -0.0010374, 0.0010858
5: 0.0080358, 0.0096861, 0.0079422, 0.0096384, -0.0010541, 0.0011518
6: 0.0092776, 0.0099004, 0.0092957, 0.0099357, -0.0005037, 0.0005925
7: -0.0194269, -0.0158444, -0.0193233, -0.0156412, -0.0024711, 0.0022128
8: 0.9681305, 0.9783950, 0.9684274, 0.9789771, -0.0071872, 0.0066239
9: 0.0040029, 0.0070196, 0.0038318, 0.0069323, -0.0018857, 0.0020905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050567
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050567
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002977, 0.0000626, -0.0002350, 0.0002654
1: -0.0000279, 0.0014503, -0.0000736, 0.0014176, -0.0009742, 0.0010466
2: 0.0141680, 0.0163817, 0.0142170, 0.0164502, -0.0015577, 0.0014445
3: 0.0000268, 0.0016914, 0.0000637, 0.0017429, -0.0011669, 0.0010799
4: -0.0043549, -0.0028195, -0.0043209, -0.0027719, -0.0011124, 0.0010488
5: 0.0079650, 0.0096265, 0.0080017, 0.0096779, -0.0011643, 0.0010773
6: 0.0093001, 0.0099272, 0.0092807, 0.0099133, -0.0005669, 0.0005583
7: -0.0192976, -0.0156905, -0.0194092, -0.0157703, -0.0022795, 0.0024819
8: 0.9685011, 0.9788356, 0.9681813, 0.9786072, -0.0067584, 0.0072822
9: 0.0038734, 0.0069107, 0.0039405, 0.0070047, -0.0021047, 0.0019376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0051567
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0051567
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002977, 0.0000628, -0.0002379, 0.0002722
1: -0.0000382, 0.0014603, -0.0000735, 0.0014179, -0.0009912, 0.0010612
2: 0.0141530, 0.0163971, 0.0142165, 0.0164501, -0.0015794, 0.0014709
3: 0.0000156, 0.0017031, 0.0000633, 0.0017429, -0.0011832, 0.0010999
4: -0.0043653, -0.0028087, -0.0043212, -0.0027720, -0.0011276, 0.0010669
5: 0.0079537, 0.0096381, 0.0080014, 0.0096779, -0.0011806, 0.0010973
6: 0.0092958, 0.0099314, 0.0092808, 0.0099134, -0.0005838, 0.0005649
7: -0.0193227, -0.0156661, -0.0194090, -0.0157696, -0.0023153, 0.0025170
8: 0.9684289, 0.9789057, 0.9681816, 0.9786091, -0.0068810, 0.0073837
9: 0.0038528, 0.0069319, 0.0039399, 0.0070046, -0.0021343, 0.0019710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0052319
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0052319
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002884, 0.0001110, -0.0002744, 0.0002495
1: -0.0000279, 0.0014503, -0.0000298, 0.0014918, -0.0010139, 0.0010041
2: 0.0141680, 0.0163817, 0.0141059, 0.0163846, -0.0014806, 0.0015036
3: 0.0000268, 0.0016914, -0.0000198, 0.0016936, -0.0011035, 0.0011247
4: -0.0043549, -0.0028195, -0.0043979, -0.0028174, -0.0011040, 0.0010941
5: 0.0079650, 0.0096265, 0.0079184, 0.0096287, -0.0011005, 0.0011221
6: 0.0093001, 0.0099272, 0.0092993, 0.0099447, -0.0005987, 0.0006278
7: -0.0192976, -0.0156905, -0.0193023, -0.0155894, -0.0023743, 0.0022950
8: 0.9685011, 0.9788356, 0.9684874, 0.9791255, -0.0070342, 0.0069358
9: 0.0038734, 0.0069107, 0.0037882, 0.0069147, -0.0019605, 0.0020184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0052457
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0052457
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002884, 0.0001111, -0.0002785, 0.0002568
1: -0.0000382, 0.0014603, -0.0000297, 0.0014920, -0.0010357, 0.0010198
2: 0.0141530, 0.0163971, 0.0141056, 0.0163845, -0.0015040, 0.0015366
3: 0.0000156, 0.0017031, -0.0000201, 0.0016936, -0.0011210, 0.0011491
4: -0.0043653, -0.0028087, -0.0043981, -0.0028175, -0.0011203, 0.0011158
5: 0.0079537, 0.0096381, 0.0079181, 0.0096287, -0.0011181, 0.0011464
6: 0.0092958, 0.0099314, 0.0092993, 0.0099448, -0.0006156, 0.0006321
7: -0.0193227, -0.0156661, -0.0193022, -0.0155889, -0.0024215, 0.0023326
8: 0.9684289, 0.9789057, 0.9684879, 0.9791269, -0.0071885, 0.0070450
9: 0.0038528, 0.0069319, 0.0037878, 0.0069146, -0.0019923, 0.0020604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0053259
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0053259
time: 0.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050005
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050005
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050322
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050322
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0049907
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0049907
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050322
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0050322
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0051559
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0051559
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052356
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052356
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052134, upper bound: 0.0052503
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052134, upper bound: 0.0052503
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052134, upper bound: 0.0053374
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052134, upper bound: 0.0053374
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052791
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052791
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053090
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053090
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052352
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0052352
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053089
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053089
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053957
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0053957
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0054927
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0049907, upper bound: 0.0054927
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052008, upper bound: 0.0054710
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052008, upper bound: 0.0054710
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052008, upper bound: 0.0055926
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052008, upper bound: 0.0055926
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050005
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050005
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050322
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050322
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0049907
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0049907
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050322
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0050322
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0051545
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0051545
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0052316
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052352, upper bound: 0.0052316
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0052456
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0052456
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0053259
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0053259
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050245
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050245
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050567
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050567
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050093
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050093
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050567
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0050567
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0051567
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0051567
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0052319
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0052353, upper bound: 0.0052319
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0052457
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0052457
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0053259
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 8, lower bound: -0.0054250, upper bound: 0.0053259

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002790, 0.0000268, -0.0001959, 0.0001959
1: 0.0000141, 0.0013627, 0.0000141, 0.0013627, -0.0008581, 0.0008581
2: 0.0142991, 0.0163189, 0.0142991, 0.0163189, -0.0012820, 0.0012820
3: 0.0001254, 0.0016442, 0.0001254, 0.0016442, -0.0009626, 0.0009626
4: -0.0042639, -0.0028630, -0.0042639, -0.0028630, -0.0008986, 0.0008986
5: 0.0080634, 0.0095794, 0.0080634, 0.0095794, -0.0009607, 0.0009607
6: 0.0093179, 0.0098900, 0.0093179, 0.0098900, -0.0003958, 0.0003958
7: -0.0191953, -0.0159042, -0.0191953, -0.0159042, -0.0020694, 0.0020694
8: 0.9687942, 0.9782236, 0.9687942, 0.9782236, -0.0059880, 0.0059880
9: 0.0040533, 0.0068246, 0.0040533, 0.0068246, -0.0017479, 0.0017479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049296, upper bound: 0.0048901
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048946, upper bound: 0.0048946
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002694, 0.0000739, -0.0002489, 0.0001949
1: 0.0000141, 0.0013627, 0.0000590, 0.0014349, -0.0009745, 0.0008797
2: 0.0142991, 0.0163189, 0.0141911, 0.0162516, -0.0013052, 0.0014563
3: 0.0001254, 0.0016442, 0.0000442, 0.0015936, -0.0009763, 0.0010937
4: -0.0042639, -0.0028630, -0.0043389, -0.0029096, -0.0009444, 0.0010195
5: 0.0080634, 0.0095794, 0.0079823, 0.0095289, -0.0009741, 0.0010916
6: 0.0093179, 0.0098900, 0.0093370, 0.0099206, -0.0004452, 0.0005039
7: -0.0191953, -0.0159042, -0.0190857, -0.0157282, -0.0023535, 0.0020652
8: 0.9687942, 0.9782236, 0.9691080, 0.9787278, -0.0068020, 0.0061062
9: 0.0040533, 0.0068246, 0.0039050, 0.0067323, -0.0017542, 0.0019872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049296, upper bound: 0.0048901
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048946, upper bound: 0.0048946
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002790, 0.0000268, -0.0002004, 0.0002046
1: 0.0000058, 0.0013729, 0.0000141, 0.0013627, -0.0008791, 0.0008798
2: 0.0142839, 0.0163313, 0.0142991, 0.0163189, -0.0013145, 0.0013133
3: 0.0001140, 0.0016536, 0.0001254, 0.0016442, -0.0009871, 0.0009862
4: -0.0042745, -0.0028544, -0.0042639, -0.0028630, -0.0009212, 0.0009226
5: 0.0080519, 0.0095887, 0.0080634, 0.0095794, -0.0009852, 0.0009842
6: 0.0093144, 0.0098943, 0.0093179, 0.0098900, -0.0004147, 0.0004051
7: -0.0192155, -0.0158793, -0.0191953, -0.0159042, -0.0021203, 0.0021224
8: 0.9687363, 0.9782947, 0.9687942, 0.9782236, -0.0061345, 0.0061399
9: 0.0040323, 0.0068416, 0.0040533, 0.0068246, -0.0017926, 0.0017908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0049227
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049228
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002694, 0.0000739, -0.0002534, 0.0002035
1: 0.0000058, 0.0013729, 0.0000590, 0.0014349, -0.0009956, 0.0009015
2: 0.0142839, 0.0163313, 0.0141911, 0.0162516, -0.0013378, 0.0014877
3: 0.0001140, 0.0016536, 0.0000442, 0.0015936, -0.0010008, 0.0011173
4: -0.0042745, -0.0028544, -0.0043389, -0.0029096, -0.0009669, 0.0010436
5: 0.0080519, 0.0095887, 0.0079823, 0.0095289, -0.0009985, 0.0011151
6: 0.0093144, 0.0098943, 0.0093370, 0.0099206, -0.0004640, 0.0005131
7: -0.0192155, -0.0158793, -0.0190857, -0.0157282, -0.0024044, 0.0021182
8: 0.9687363, 0.9782947, 0.9691080, 0.9787278, -0.0069484, 0.0062581
9: 0.0040323, 0.0068416, 0.0039050, 0.0067323, -0.0017988, 0.0020301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0049227
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049228
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002808, 0.0000335, -0.0002046, 0.0002004
1: 0.0000141, 0.0013627, 0.0000058, 0.0013729, -0.0008798, 0.0008791
2: 0.0142991, 0.0163189, 0.0142839, 0.0163313, -0.0013133, 0.0013145
3: 0.0001254, 0.0016442, 0.0001140, 0.0016536, -0.0009862, 0.0009871
4: -0.0042639, -0.0028630, -0.0042745, -0.0028544, -0.0009226, 0.0009212
5: 0.0080634, 0.0095794, 0.0080519, 0.0095887, -0.0009842, 0.0009852
6: 0.0093179, 0.0098900, 0.0093144, 0.0098943, -0.0004051, 0.0004147
7: -0.0191953, -0.0159042, -0.0192155, -0.0158793, -0.0021224, 0.0021203
8: 0.9687942, 0.9782236, 0.9687363, 0.9782947, -0.0061399, 0.0061345
9: 0.0040533, 0.0068246, 0.0040323, 0.0068416, -0.0017908, 0.0017926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049502, upper bound: 0.0048592
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049228, upper bound: 0.0048668
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002714, 0.0000805, -0.0002554, 0.0001975
1: 0.0000141, 0.0013627, 0.0000494, 0.0014451, -0.0009853, 0.0008966
2: 0.0142991, 0.0163189, 0.0141759, 0.0162660, -0.0013311, 0.0014724
3: 0.0001254, 0.0016442, 0.0000328, 0.0016044, -0.0009958, 0.0011058
4: -0.0042639, -0.0028630, -0.0043494, -0.0028997, -0.0009625, 0.0010307
5: 0.0080634, 0.0095794, 0.0079709, 0.0095397, -0.0009935, 0.0011037
6: 0.0093179, 0.0098900, 0.0093329, 0.0099249, -0.0004498, 0.0005204
7: -0.0191953, -0.0159042, -0.0191090, -0.0157033, -0.0023797, 0.0020988
8: 0.9687942, 0.9782236, 0.9690413, 0.9787990, -0.0068772, 0.0062265
9: 0.0040533, 0.0068246, 0.0038841, 0.0067519, -0.0017861, 0.0020093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049502, upper bound: 0.0048592
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049228, upper bound: 0.0048668
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002808, 0.0000335, -0.0002034, 0.0002034
1: 0.0000058, 0.0013729, 0.0000058, 0.0013729, -0.0008787, 0.0008787
2: 0.0142839, 0.0163313, 0.0142839, 0.0163313, -0.0013119, 0.0013119
3: 0.0001140, 0.0016536, 0.0001140, 0.0016536, -0.0009846, 0.0009846
4: -0.0042745, -0.0028544, -0.0042745, -0.0028544, -0.0009228, 0.0009228
5: 0.0080519, 0.0095887, 0.0080519, 0.0095887, -0.0009826, 0.0009826
6: 0.0093144, 0.0098943, 0.0093144, 0.0098943, -0.0004180, 0.0004180
7: -0.0192155, -0.0158793, -0.0192155, -0.0158793, -0.0021114, 0.0021114
8: 0.9687363, 0.9782947, 0.9687363, 0.9782947, -0.0061287, 0.0061287
9: 0.0040323, 0.0068416, 0.0040323, 0.0068416, -0.0017851, 0.0017851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0049227
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049228
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002714, 0.0000805, -0.0002561, 0.0002025
1: 0.0000058, 0.0013729, 0.0000494, 0.0014451, -0.0009951, 0.0009028
2: 0.0142839, 0.0163313, 0.0141759, 0.0162660, -0.0013387, 0.0014862
3: 0.0001140, 0.0016536, 0.0000328, 0.0016044, -0.0010011, 0.0011157
4: -0.0042745, -0.0028544, -0.0043494, -0.0028997, -0.0009713, 0.0010437
5: 0.0080519, 0.0095887, 0.0079709, 0.0095397, -0.0009987, 0.0011135
6: 0.0093144, 0.0098943, 0.0093329, 0.0099249, -0.0004674, 0.0005293
7: -0.0192155, -0.0158793, -0.0191090, -0.0157033, -0.0023955, 0.0021097
8: 0.9687363, 0.9782947, 0.9690413, 0.9787990, -0.0069427, 0.0062630
9: 0.0040323, 0.0068416, 0.0038841, 0.0067519, -0.0017939, 0.0020244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0049227
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049228
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002790, 0.0000268, -0.0001949, 0.0002489
1: 0.0000590, 0.0014349, 0.0000141, 0.0013627, -0.0008797, 0.0009745
2: 0.0141911, 0.0162516, 0.0142991, 0.0163189, -0.0014563, 0.0013052
3: 0.0000442, 0.0015936, 0.0001254, 0.0016442, -0.0010937, 0.0009763
4: -0.0043389, -0.0029096, -0.0042639, -0.0028630, -0.0010195, 0.0009444
5: 0.0079823, 0.0095289, 0.0080634, 0.0095794, -0.0010916, 0.0009741
6: 0.0093370, 0.0099206, 0.0093179, 0.0098900, -0.0005039, 0.0004452
7: -0.0190857, -0.0157282, -0.0191953, -0.0159042, -0.0020652, 0.0023535
8: 0.9691080, 0.9787278, 0.9687942, 0.9782236, -0.0061062, 0.0068020
9: 0.0039050, 0.0067323, 0.0040533, 0.0068246, -0.0019872, 0.0017542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049192, upper bound: 0.0050613
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048880, upper bound: 0.0050613
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002808, 0.0000335, -0.0002035, 0.0002534
1: 0.0000590, 0.0014349, 0.0000058, 0.0013729, -0.0009015, 0.0009956
2: 0.0141911, 0.0162516, 0.0142839, 0.0163313, -0.0014877, 0.0013378
3: 0.0000442, 0.0015936, 0.0001140, 0.0016536, -0.0011173, 0.0010008
4: -0.0043389, -0.0029096, -0.0042745, -0.0028544, -0.0010436, 0.0009669
5: 0.0079823, 0.0095289, 0.0080519, 0.0095887, -0.0011151, 0.0009985
6: 0.0093370, 0.0099206, 0.0093144, 0.0098943, -0.0005131, 0.0004640
7: -0.0190857, -0.0157282, -0.0192155, -0.0158793, -0.0021182, 0.0024044
8: 0.9691080, 0.9787278, 0.9687363, 0.9782947, -0.0062581, 0.0069484
9: 0.0039050, 0.0067323, 0.0040323, 0.0068416, -0.0020301, 0.0017988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049192, upper bound: 0.0050613
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048880, upper bound: 0.0050613
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002790, 0.0000268, -0.0001975, 0.0002554
1: 0.0000494, 0.0014451, 0.0000141, 0.0013627, -0.0008966, 0.0009853
2: 0.0141759, 0.0162660, 0.0142991, 0.0163189, -0.0014724, 0.0013311
3: 0.0000328, 0.0016044, 0.0001254, 0.0016442, -0.0011058, 0.0009958
4: -0.0043494, -0.0028997, -0.0042639, -0.0028630, -0.0010307, 0.0009625
5: 0.0079709, 0.0095397, 0.0080634, 0.0095794, -0.0011037, 0.0009935
6: 0.0093329, 0.0099249, 0.0093179, 0.0098900, -0.0005204, 0.0004498
7: -0.0191090, -0.0157033, -0.0191953, -0.0159042, -0.0020988, 0.0023797
8: 0.9690413, 0.9787990, 0.9687942, 0.9782236, -0.0062265, 0.0068772
9: 0.0038841, 0.0067519, 0.0040533, 0.0068246, -0.0020093, 0.0017861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0051343
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0051355
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002808, 0.0000335, -0.0002025, 0.0002561
1: 0.0000494, 0.0014451, 0.0000058, 0.0013729, -0.0009028, 0.0009951
2: 0.0141759, 0.0162660, 0.0142839, 0.0163313, -0.0014862, 0.0013387
3: 0.0000328, 0.0016044, 0.0001140, 0.0016536, -0.0011157, 0.0010011
4: -0.0043494, -0.0028997, -0.0042745, -0.0028544, -0.0010437, 0.0009713
5: 0.0079709, 0.0095397, 0.0080519, 0.0095887, -0.0011135, 0.0009987
6: 0.0093329, 0.0099249, 0.0093144, 0.0098943, -0.0005293, 0.0004674
7: -0.0191090, -0.0157033, -0.0192155, -0.0158793, -0.0021097, 0.0023955
8: 0.9690413, 0.9787990, 0.9687363, 0.9782947, -0.0062630, 0.0069427
9: 0.0038841, 0.0067519, 0.0040323, 0.0068416, -0.0020244, 0.0017939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0051343
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0051355
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002694, 0.0000739, -0.0002318, 0.0002318
1: 0.0000590, 0.0014349, 0.0000590, 0.0014349, -0.0009143, 0.0009143
2: 0.0141911, 0.0162516, 0.0141911, 0.0162516, -0.0013578, 0.0013578
3: 0.0000442, 0.0015936, 0.0000442, 0.0015936, -0.0010163, 0.0010163
4: -0.0043389, -0.0029096, -0.0043389, -0.0029096, -0.0009826, 0.0009826
5: 0.0079823, 0.0095289, 0.0079823, 0.0095289, -0.0010140, 0.0010140
6: 0.0093370, 0.0099206, 0.0093370, 0.0099206, -0.0005313, 0.0005313
7: -0.0190857, -0.0157282, -0.0190857, -0.0157282, -0.0021529, 0.0021530
8: 0.9691080, 0.9787278, 0.9691080, 0.9787278, -0.0063497, 0.0063497
9: 0.0039050, 0.0067323, 0.0039050, 0.0067323, -0.0018283, 0.0018283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051452, upper bound: 0.0051497
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051276, upper bound: 0.0051504
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002714, 0.0000805, -0.0002395, 0.0002359
1: 0.0000590, 0.0014349, 0.0000494, 0.0014451, -0.0009321, 0.0009375
2: 0.0141911, 0.0162516, 0.0141759, 0.0162660, -0.0013921, 0.0013844
3: 0.0000442, 0.0015936, 0.0000328, 0.0016044, -0.0010415, 0.0010363
4: -0.0043389, -0.0029096, -0.0043494, -0.0028997, -0.0010066, 0.0010011
5: 0.0079823, 0.0095289, 0.0079709, 0.0095397, -0.0010391, 0.0010340
6: 0.0093370, 0.0099206, 0.0093329, 0.0099249, -0.0005389, 0.0005484
7: -0.0190857, -0.0157282, -0.0191090, -0.0157033, -0.0021963, 0.0021986
8: 0.9691080, 0.9787278, 0.9690413, 0.9787990, -0.0064740, 0.0065113
9: 0.0039050, 0.0067323, 0.0038841, 0.0067519, -0.0018695, 0.0018649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051452, upper bound: 0.0051497
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051276, upper bound: 0.0051504
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002694, 0.0000739, -0.0002359, 0.0002395
1: 0.0000494, 0.0014451, 0.0000590, 0.0014349, -0.0009375, 0.0009321
2: 0.0141759, 0.0162660, 0.0141911, 0.0162516, -0.0013844, 0.0013921
3: 0.0000328, 0.0016044, 0.0000442, 0.0015936, -0.0010363, 0.0010415
4: -0.0043494, -0.0028997, -0.0043389, -0.0029096, -0.0010011, 0.0010066
5: 0.0079709, 0.0095397, 0.0079823, 0.0095289, -0.0010340, 0.0010391
6: 0.0093329, 0.0099249, 0.0093370, 0.0099206, -0.0005484, 0.0005389
7: -0.0191090, -0.0157033, -0.0190857, -0.0157282, -0.0021986, 0.0021963
8: 0.9690413, 0.9787990, 0.9691080, 0.9787278, -0.0065113, 0.0064740
9: 0.0038841, 0.0067519, 0.0039050, 0.0067323, -0.0018649, 0.0018695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051301, upper bound: 0.0052356
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051094, upper bound: 0.0052382
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002714, 0.0000805, -0.0002395, 0.0002395
1: 0.0000494, 0.0014451, 0.0000494, 0.0014451, -0.0009394, 0.0009394
2: 0.0141759, 0.0162660, 0.0141759, 0.0162660, -0.0013929, 0.0013929
3: 0.0000328, 0.0016044, 0.0000328, 0.0016044, -0.0010417, 0.0010417
4: -0.0043494, -0.0028997, -0.0043494, -0.0028997, -0.0010120, 0.0010120
5: 0.0079709, 0.0095397, 0.0079709, 0.0095397, -0.0010393, 0.0010393
6: 0.0093329, 0.0099249, 0.0093329, 0.0099249, -0.0005584, 0.0005584
7: -0.0191090, -0.0157033, -0.0191090, -0.0157033, -0.0021980, 0.0021980
8: 0.9690413, 0.9787990, 0.9690413, 0.9787990, -0.0065165, 0.0065165
9: 0.0038841, 0.0067519, 0.0038841, 0.0067519, -0.0018687, 0.0018687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051301, upper bound: 0.0052356
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051094, upper bound: 0.0052382
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002973, 0.0000352, -0.0002130, 0.0002241
1: 0.0000141, 0.0013627, -0.0000716, 0.0013756, -0.0009148, 0.0009902
2: 0.0142991, 0.0163189, 0.0142798, 0.0164472, -0.0014800, 0.0013669
3: 0.0001254, 0.0016442, 0.0001109, 0.0017407, -0.0011115, 0.0010265
4: -0.0042639, -0.0028630, -0.0042773, -0.0027740, -0.0010353, 0.0009575
5: 0.0080634, 0.0095794, 0.0080489, 0.0096757, -0.0011094, 0.0010245
6: 0.0093179, 0.0098900, 0.0092816, 0.0098955, -0.0004199, 0.0004509
7: -0.0191953, -0.0159042, -0.0194043, -0.0158728, -0.0022078, 0.0023922
8: 0.9687942, 0.9782236, 0.9681953, 0.9783136, -0.0063847, 0.0069124
9: 0.0040533, 0.0068246, 0.0040268, 0.0070006, -0.0020198, 0.0018645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049296, upper bound: 0.0051920
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048946, upper bound: 0.0051920
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002880, 0.0000840, -0.0002638, 0.0002190
1: 0.0000141, 0.0013627, -0.0000279, 0.0014503, -0.0010136, 0.0009900
2: 0.0142991, 0.0163189, 0.0141680, 0.0163817, -0.0014721, 0.0015149
3: 0.0001254, 0.0016442, 0.0000268, 0.0016914, -0.0011022, 0.0011378
4: -0.0042639, -0.0028630, -0.0043549, -0.0028195, -0.0010547, 0.0010602
5: 0.0080634, 0.0095794, 0.0079650, 0.0096265, -0.0010997, 0.0011356
6: 0.0093179, 0.0098900, 0.0093001, 0.0099272, -0.0004618, 0.0005464
7: -0.0191953, -0.0159042, -0.0192976, -0.0156905, -0.0024489, 0.0023407
8: 0.9687942, 0.9782236, 0.9685011, 0.9788356, -0.0070755, 0.0068834
9: 0.0040533, 0.0068246, 0.0038734, 0.0069107, -0.0019854, 0.0020676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049296, upper bound: 0.0051920
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048946, upper bound: 0.0051920
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002973, 0.0000352, -0.0002174, 0.0002328
1: 0.0000058, 0.0013729, -0.0000716, 0.0013756, -0.0009359, 0.0010119
2: 0.0142839, 0.0163313, 0.0142798, 0.0164472, -0.0015125, 0.0013983
3: 0.0001140, 0.0016536, 0.0001109, 0.0017407, -0.0011360, 0.0010501
4: -0.0042745, -0.0028544, -0.0042773, -0.0027740, -0.0010579, 0.0009816
5: 0.0080519, 0.0095887, 0.0080489, 0.0096757, -0.0011338, 0.0010480
6: 0.0093144, 0.0098943, 0.0092816, 0.0098955, -0.0004387, 0.0004601
7: -0.0192155, -0.0158793, -0.0194043, -0.0158728, -0.0022588, 0.0024452
8: 0.9687363, 0.9782947, 0.9681953, 0.9783136, -0.0065312, 0.0070643
9: 0.0040323, 0.0068416, 0.0040268, 0.0070006, -0.0020645, 0.0019074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0052200
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0052200
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002880, 0.0000840, -0.0002682, 0.0002276
1: 0.0000058, 0.0013729, -0.0000279, 0.0014503, -0.0010347, 0.0010118
2: 0.0142839, 0.0163313, 0.0141680, 0.0163817, -0.0015046, 0.0015463
3: 0.0001140, 0.0016536, 0.0000268, 0.0016914, -0.0011266, 0.0011613
4: -0.0042745, -0.0028544, -0.0043549, -0.0028195, -0.0010773, 0.0010842
5: 0.0080519, 0.0095887, 0.0079650, 0.0096265, -0.0011241, 0.0011591
6: 0.0093144, 0.0098943, 0.0093001, 0.0099272, -0.0004806, 0.0005556
7: -0.0192155, -0.0158793, -0.0192976, -0.0156905, -0.0024999, 0.0023938
8: 0.9687363, 0.9782947, 0.9685011, 0.9788356, -0.0072219, 0.0070353
9: 0.0040323, 0.0068416, 0.0038734, 0.0069107, -0.0020300, 0.0021104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0052203
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0052203
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002993, 0.0000428, -0.0002200, 0.0002262
1: 0.0000141, 0.0013627, -0.0000809, 0.0013873, -0.0009262, 0.0010042
2: 0.0142991, 0.0163189, 0.0142624, 0.0164611, -0.0014994, 0.0013840
3: 0.0001254, 0.0016442, 0.0000978, 0.0017511, -0.0011255, 0.0010393
4: -0.0042639, -0.0028630, -0.0042894, -0.0027644, -0.0010538, 0.0009693
5: 0.0080634, 0.0095794, 0.0080358, 0.0096861, -0.0011233, 0.0010373
6: 0.0093179, 0.0098900, 0.0092776, 0.0099004, -0.0004247, 0.0004692
7: -0.0191953, -0.0159042, -0.0194269, -0.0158444, -0.0022356, 0.0024167
8: 0.9687942, 0.9782236, 0.9681305, 0.9783950, -0.0064643, 0.0070046
9: 0.0040533, 0.0068246, 0.0040029, 0.0070196, -0.0020419, 0.0018879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049502, upper bound: 0.0051443
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049228, upper bound: 0.0051443
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000268, -0.0002902, 0.0000905, -0.0002696, 0.0002217
1: 0.0000141, 0.0013627, -0.0000382, 0.0014603, -0.0010246, 0.0010091
2: 0.0142991, 0.0163189, 0.0141530, 0.0163971, -0.0014998, 0.0015313
3: 0.0001254, 0.0016442, 0.0000156, 0.0017031, -0.0011227, 0.0011501
4: -0.0042639, -0.0028630, -0.0043653, -0.0028087, -0.0010771, 0.0010715
5: 0.0080634, 0.0095794, 0.0079537, 0.0096381, -0.0011202, 0.0011479
6: 0.0093179, 0.0098900, 0.0092958, 0.0099314, -0.0004665, 0.0005624
7: -0.0191953, -0.0159042, -0.0193227, -0.0156661, -0.0024757, 0.0023744
8: 0.9687942, 0.9782236, 0.9684289, 0.9789057, -0.0071521, 0.0070134
9: 0.0040533, 0.0068246, 0.0038528, 0.0069319, -0.0020175, 0.0020901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049502, upper bound: 0.0051443
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049228, upper bound: 0.0051443
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002993, 0.0000428, -0.0002207, 0.0002316
1: 0.0000058, 0.0013729, -0.0000809, 0.0013873, -0.0009351, 0.0010096
2: 0.0142839, 0.0163313, 0.0142624, 0.0164611, -0.0015084, 0.0013964
3: 0.0001140, 0.0016536, 0.0000978, 0.0017511, -0.0011326, 0.0010482
4: -0.0042745, -0.0028544, -0.0042894, -0.0027644, -0.0010574, 0.0009814
5: 0.0080519, 0.0095887, 0.0080358, 0.0096861, -0.0011303, 0.0010461
6: 0.0093144, 0.0098943, 0.0092776, 0.0099004, -0.0004419, 0.0004723
7: -0.0192155, -0.0158793, -0.0194269, -0.0158444, -0.0022492, 0.0024337
8: 0.9687363, 0.9782947, 0.9681305, 0.9783950, -0.0065233, 0.0070457
9: 0.0040323, 0.0068416, 0.0040029, 0.0070196, -0.0020561, 0.0019011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0052200
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0052200
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002808, 0.0000335, -0.0002902, 0.0000905, -0.0002710, 0.0002263
1: 0.0000058, 0.0013729, -0.0000382, 0.0014603, -0.0010340, 0.0010103
2: 0.0142839, 0.0163313, 0.0141530, 0.0163971, -0.0015010, 0.0015445
3: 0.0001140, 0.0016536, 0.0000156, 0.0017031, -0.0011234, 0.0011595
4: -0.0042745, -0.0028544, -0.0043653, -0.0028087, -0.0010781, 0.0010841
5: 0.0080519, 0.0095887, 0.0079537, 0.0096381, -0.0011208, 0.0011572
6: 0.0093144, 0.0098943, 0.0092958, 0.0099314, -0.0004839, 0.0005705
7: -0.0192155, -0.0158793, -0.0193227, -0.0156661, -0.0024904, 0.0023803
8: 0.9687363, 0.9782947, 0.9684289, 0.9789057, -0.0072145, 0.0070200
9: 0.0040323, 0.0068416, 0.0038528, 0.0069319, -0.0020201, 0.0021042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0052203
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0052203
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002973, 0.0000352, -0.0002119, 0.0002771
1: 0.0000590, 0.0014349, -0.0000716, 0.0013756, -0.0009365, 0.0011066
2: 0.0141911, 0.0162516, 0.0142798, 0.0164472, -0.0016543, 0.0013902
3: 0.0000442, 0.0015936, 0.0001109, 0.0017407, -0.0012426, 0.0010402
4: -0.0043389, -0.0029096, -0.0042773, -0.0027740, -0.0011562, 0.0010033
5: 0.0079823, 0.0095289, 0.0080489, 0.0096757, -0.0012402, 0.0010379
6: 0.0093370, 0.0099206, 0.0092816, 0.0098955, -0.0005280, 0.0005003
7: -0.0190857, -0.0157282, -0.0194043, -0.0158728, -0.0022036, 0.0026763
8: 0.9691080, 0.9787278, 0.9681953, 0.9783136, -0.0065028, 0.0077263
9: 0.0039050, 0.0067323, 0.0040268, 0.0070006, -0.0022590, 0.0018708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049192, upper bound: 0.0053095
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048880, upper bound: 0.0053095
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002993, 0.0000428, -0.0002189, 0.0002792
1: 0.0000590, 0.0014349, -0.0000809, 0.0013873, -0.0009479, 0.0011206
2: 0.0141911, 0.0162516, 0.0142624, 0.0164611, -0.0016737, 0.0014073
3: 0.0000442, 0.0015936, 0.0000978, 0.0017511, -0.0012566, 0.0010530
4: -0.0043389, -0.0029096, -0.0042894, -0.0027644, -0.0011747, 0.0010151
5: 0.0079823, 0.0095289, 0.0080358, 0.0096861, -0.0012542, 0.0010507
6: 0.0093370, 0.0099206, 0.0092776, 0.0099004, -0.0005328, 0.0005186
7: -0.0190857, -0.0157282, -0.0194269, -0.0158444, -0.0022314, 0.0027008
8: 0.9691080, 0.9787278, 0.9681305, 0.9783950, -0.0065825, 0.0078186
9: 0.0039050, 0.0067323, 0.0040029, 0.0070196, -0.0022812, 0.0018942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049192, upper bound: 0.0053095
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048880, upper bound: 0.0053095
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002973, 0.0000352, -0.0002145, 0.0002836
1: 0.0000494, 0.0014451, -0.0000716, 0.0013756, -0.0009534, 0.0011173
2: 0.0141759, 0.0162660, 0.0142798, 0.0164472, -0.0016704, 0.0014161
3: 0.0000328, 0.0016044, 0.0001109, 0.0017407, -0.0012547, 0.0010597
4: -0.0043494, -0.0028997, -0.0042773, -0.0027740, -0.0011674, 0.0010215
5: 0.0079709, 0.0095397, 0.0080489, 0.0096757, -0.0012523, 0.0010573
6: 0.0093329, 0.0099249, 0.0092816, 0.0098955, -0.0005444, 0.0005049
7: -0.0191090, -0.0157033, -0.0194043, -0.0158728, -0.0022373, 0.0027025
8: 0.9690413, 0.9787990, 0.9681953, 0.9783136, -0.0066232, 0.0078015
9: 0.0038841, 0.0067519, 0.0040268, 0.0070006, -0.0022811, 0.0019027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0054052
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0054055
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002993, 0.0000428, -0.0002199, 0.0002844
1: 0.0000494, 0.0014451, -0.0000809, 0.0013873, -0.0009592, 0.0011260
2: 0.0141759, 0.0162660, 0.0142624, 0.0164611, -0.0016827, 0.0014232
3: 0.0000328, 0.0016044, 0.0000978, 0.0017511, -0.0012637, 0.0010646
4: -0.0043494, -0.0028997, -0.0042894, -0.0027644, -0.0011784, 0.0010299
5: 0.0079709, 0.0095397, 0.0080358, 0.0096861, -0.0012612, 0.0010621
6: 0.0093329, 0.0099249, 0.0092776, 0.0099004, -0.0005533, 0.0005217
7: -0.0191090, -0.0157033, -0.0194269, -0.0158444, -0.0022475, 0.0027178
8: 0.9690413, 0.9787990, 0.9681305, 0.9783950, -0.0066577, 0.0078597
9: 0.0038841, 0.0067519, 0.0040029, 0.0070196, -0.0022953, 0.0019099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0054052
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0054055
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002880, 0.0000840, -0.0002494, 0.0002591
1: 0.0000590, 0.0014349, -0.0000279, 0.0014503, -0.0009697, 0.0010350
2: 0.0141911, 0.0162516, 0.0141680, 0.0163817, -0.0015415, 0.0014407
3: 0.0000442, 0.0015936, 0.0000268, 0.0016914, -0.0011554, 0.0010787
4: -0.0043389, -0.0029096, -0.0043549, -0.0028195, -0.0011002, 0.0010401
5: 0.0079823, 0.0095289, 0.0079650, 0.0096265, -0.0011530, 0.0010763
6: 0.0093370, 0.0099206, 0.0093001, 0.0099272, -0.0005548, 0.0005728
7: -0.0190857, -0.0157282, -0.0192976, -0.0156905, -0.0022881, 0.0024626
8: 0.9691080, 0.9787278, 0.9685011, 0.9788356, -0.0067370, 0.0072048
9: 0.0039050, 0.0067323, 0.0038734, 0.0069107, -0.0020870, 0.0019422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051380, upper bound: 0.0053837
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051185, upper bound: 0.0053842
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002694, 0.0000739, -0.0002902, 0.0000905, -0.0002556, 0.0002614
1: 0.0000590, 0.0014349, -0.0000382, 0.0014603, -0.0009806, 0.0010561
2: 0.0141911, 0.0162516, 0.0141530, 0.0163971, -0.0015700, 0.0014571
3: 0.0000442, 0.0015936, 0.0000156, 0.0017031, -0.0011755, 0.0010910
4: -0.0043389, -0.0029096, -0.0043653, -0.0028087, -0.0011253, 0.0010515
5: 0.0079823, 0.0095289, 0.0079537, 0.0096381, -0.0011729, 0.0010886
6: 0.0093370, 0.0099206, 0.0092958, 0.0099314, -0.0005595, 0.0005891
7: -0.0190857, -0.0157282, -0.0193227, -0.0156661, -0.0023149, 0.0024912
8: 0.9691080, 0.9787278, 0.9684289, 0.9789057, -0.0068136, 0.0073413
9: 0.0039050, 0.0067323, 0.0038528, 0.0069319, -0.0021156, 0.0019647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051380, upper bound: 0.0053837
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051185, upper bound: 0.0053842
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002880, 0.0000840, -0.0002534, 0.0002667
1: 0.0000494, 0.0014451, -0.0000279, 0.0014503, -0.0009929, 0.0010528
2: 0.0141759, 0.0162660, 0.0141680, 0.0163817, -0.0015681, 0.0014751
3: 0.0000328, 0.0016044, 0.0000268, 0.0016914, -0.0011755, 0.0011039
4: -0.0043494, -0.0028997, -0.0043549, -0.0028195, -0.0011187, 0.0010641
5: 0.0079709, 0.0095397, 0.0079650, 0.0096265, -0.0011730, 0.0011014
6: 0.0093329, 0.0099249, 0.0093001, 0.0099272, -0.0005719, 0.0005803
7: -0.0191090, -0.0157033, -0.0192976, -0.0156905, -0.0023337, 0.0025060
8: 0.9690413, 0.9787990, 0.9685011, 0.9788356, -0.0068986, 0.0073291
9: 0.0038841, 0.0067519, 0.0038734, 0.0069107, -0.0021236, 0.0019833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051213, upper bound: 0.0055035
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050970, upper bound: 0.0055046
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0000805, -0.0002902, 0.0000905, -0.0002570, 0.0002666
1: 0.0000494, 0.0014451, -0.0000382, 0.0014603, -0.0009945, 0.0010564
2: 0.0141759, 0.0162660, 0.0141530, 0.0163971, -0.0015710, 0.0014755
3: 0.0000328, 0.0016044, 0.0000156, 0.0017031, -0.0011772, 0.0011038
4: -0.0043494, -0.0028997, -0.0043653, -0.0028087, -0.0011266, 0.0010693
5: 0.0079709, 0.0095397, 0.0079537, 0.0096381, -0.0011747, 0.0011013
6: 0.0093329, 0.0099249, 0.0092958, 0.0099314, -0.0005818, 0.0005981
7: -0.0191090, -0.0157033, -0.0193227, -0.0156661, -0.0023326, 0.0025049
8: 0.9690413, 0.9787990, 0.9684289, 0.9789057, -0.0069023, 0.0073454
9: 0.0038841, 0.0067519, 0.0038528, 0.0069319, -0.0021240, 0.0019821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051213, upper bound: 0.0055035
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050970, upper bound: 0.0055046
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002790, 0.0000268, -0.0002241, 0.0002130
1: -0.0000716, 0.0013756, 0.0000141, 0.0013627, -0.0009902, 0.0009148
2: 0.0142798, 0.0164472, 0.0142991, 0.0163189, -0.0013669, 0.0014800
3: 0.0001109, 0.0017407, 0.0001254, 0.0016442, -0.0010265, 0.0011115
4: -0.0042773, -0.0027740, -0.0042639, -0.0028630, -0.0009575, 0.0010353
5: 0.0080489, 0.0096757, 0.0080634, 0.0095794, -0.0010245, 0.0011094
6: 0.0092816, 0.0098955, 0.0093179, 0.0098900, -0.0004509, 0.0004199
7: -0.0194043, -0.0158728, -0.0191953, -0.0159042, -0.0023922, 0.0022078
8: 0.9681953, 0.9783136, 0.9687942, 0.9782236, -0.0069124, 0.0063847
9: 0.0040268, 0.0070006, 0.0040533, 0.0068246, -0.0018645, 0.0020198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051949, upper bound: 0.0048901
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0048946
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002694, 0.0000739, -0.0002771, 0.0002119
1: -0.0000716, 0.0013756, 0.0000590, 0.0014349, -0.0011066, 0.0009365
2: 0.0142798, 0.0164472, 0.0141911, 0.0162516, -0.0013902, 0.0016543
3: 0.0001109, 0.0017407, 0.0000442, 0.0015936, -0.0010402, 0.0012426
4: -0.0042773, -0.0027740, -0.0043389, -0.0029096, -0.0010033, 0.0011562
5: 0.0080489, 0.0096757, 0.0079823, 0.0095289, -0.0010379, 0.0012402
6: 0.0092816, 0.0098955, 0.0093370, 0.0099206, -0.0005003, 0.0005280
7: -0.0194043, -0.0158728, -0.0190857, -0.0157282, -0.0026763, 0.0022036
8: 0.9681953, 0.9783136, 0.9691080, 0.9787278, -0.0077263, 0.0065028
9: 0.0040268, 0.0070006, 0.0039050, 0.0067323, -0.0018708, 0.0022590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051949, upper bound: 0.0048901
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0048946
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002790, 0.0000268, -0.0002262, 0.0002200
1: -0.0000809, 0.0013873, 0.0000141, 0.0013627, -0.0010042, 0.0009262
2: 0.0142624, 0.0164611, 0.0142991, 0.0163189, -0.0013840, 0.0014994
3: 0.0000978, 0.0017511, 0.0001254, 0.0016442, -0.0010393, 0.0011255
4: -0.0042894, -0.0027644, -0.0042639, -0.0028630, -0.0009693, 0.0010538
5: 0.0080358, 0.0096861, 0.0080634, 0.0095794, -0.0010373, 0.0011233
6: 0.0092776, 0.0099004, 0.0093179, 0.0098900, -0.0004692, 0.0004247
7: -0.0194269, -0.0158444, -0.0191953, -0.0159042, -0.0024167, 0.0022356
8: 0.9681305, 0.9783950, 0.9687942, 0.9782236, -0.0070046, 0.0064643
9: 0.0040029, 0.0070196, 0.0040533, 0.0068246, -0.0018879, 0.0020419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0049227
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002694, 0.0000739, -0.0002792, 0.0002189
1: -0.0000809, 0.0013873, 0.0000590, 0.0014349, -0.0011206, 0.0009479
2: 0.0142624, 0.0164611, 0.0141911, 0.0162516, -0.0014073, 0.0016737
3: 0.0000978, 0.0017511, 0.0000442, 0.0015936, -0.0010530, 0.0012566
4: -0.0042894, -0.0027644, -0.0043389, -0.0029096, -0.0010151, 0.0011747
5: 0.0080358, 0.0096861, 0.0079823, 0.0095289, -0.0010507, 0.0012542
6: 0.0092776, 0.0099004, 0.0093370, 0.0099206, -0.0005186, 0.0005328
7: -0.0194269, -0.0158444, -0.0190857, -0.0157282, -0.0027008, 0.0022314
8: 0.9681305, 0.9783950, 0.9691080, 0.9787278, -0.0078186, 0.0065825
9: 0.0040029, 0.0070196, 0.0039050, 0.0067323, -0.0018942, 0.0022812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0049227
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002808, 0.0000335, -0.0002328, 0.0002174
1: -0.0000716, 0.0013756, 0.0000058, 0.0013729, -0.0010119, 0.0009359
2: 0.0142798, 0.0164472, 0.0142839, 0.0163313, -0.0013983, 0.0015125
3: 0.0001109, 0.0017407, 0.0001140, 0.0016536, -0.0010501, 0.0011360
4: -0.0042773, -0.0027740, -0.0042745, -0.0028544, -0.0009816, 0.0010579
5: 0.0080489, 0.0096757, 0.0080519, 0.0095887, -0.0010480, 0.0011338
6: 0.0092816, 0.0098955, 0.0093144, 0.0098943, -0.0004601, 0.0004387
7: -0.0194043, -0.0158728, -0.0192155, -0.0158793, -0.0024452, 0.0022588
8: 0.9681953, 0.9783136, 0.9687363, 0.9782947, -0.0070643, 0.0065312
9: 0.0040268, 0.0070006, 0.0040323, 0.0068416, -0.0019074, 0.0020645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052207, upper bound: 0.0048592
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052200, upper bound: 0.0048668
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002714, 0.0000805, -0.0002836, 0.0002145
1: -0.0000716, 0.0013756, 0.0000494, 0.0014451, -0.0011173, 0.0009534
2: 0.0142798, 0.0164472, 0.0141759, 0.0162660, -0.0014161, 0.0016704
3: 0.0001109, 0.0017407, 0.0000328, 0.0016044, -0.0010597, 0.0012547
4: -0.0042773, -0.0027740, -0.0043494, -0.0028997, -0.0010215, 0.0011674
5: 0.0080489, 0.0096757, 0.0079709, 0.0095397, -0.0010573, 0.0012523
6: 0.0092816, 0.0098955, 0.0093329, 0.0099249, -0.0005049, 0.0005444
7: -0.0194043, -0.0158728, -0.0191090, -0.0157033, -0.0027025, 0.0022373
8: 0.9681953, 0.9783136, 0.9690413, 0.9787990, -0.0078015, 0.0066232
9: 0.0040268, 0.0070006, 0.0038841, 0.0067519, -0.0019027, 0.0022811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052207, upper bound: 0.0048592
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052200, upper bound: 0.0048668
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002808, 0.0000335, -0.0002316, 0.0002207
1: -0.0000809, 0.0013873, 0.0000058, 0.0013729, -0.0010096, 0.0009351
2: 0.0142624, 0.0164611, 0.0142839, 0.0163313, -0.0013964, 0.0015084
3: 0.0000978, 0.0017511, 0.0001140, 0.0016536, -0.0010482, 0.0011326
4: -0.0042894, -0.0027644, -0.0042745, -0.0028544, -0.0009814, 0.0010574
5: 0.0080358, 0.0096861, 0.0080519, 0.0095887, -0.0010461, 0.0011303
6: 0.0092776, 0.0099004, 0.0093144, 0.0098943, -0.0004723, 0.0004419
7: -0.0194269, -0.0158444, -0.0192155, -0.0158793, -0.0024337, 0.0022492
8: 0.9681305, 0.9783950, 0.9687363, 0.9782947, -0.0070457, 0.0065233
9: 0.0040029, 0.0070196, 0.0040323, 0.0068416, -0.0019011, 0.0020561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0049227
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002714, 0.0000805, -0.0002844, 0.0002199
1: -0.0000809, 0.0013873, 0.0000494, 0.0014451, -0.0011260, 0.0009592
2: 0.0142624, 0.0164611, 0.0141759, 0.0162660, -0.0014232, 0.0016827
3: 0.0000978, 0.0017511, 0.0000328, 0.0016044, -0.0010646, 0.0012637
4: -0.0042894, -0.0027644, -0.0043494, -0.0028997, -0.0010299, 0.0011784
5: 0.0080358, 0.0096861, 0.0079709, 0.0095397, -0.0010621, 0.0012612
6: 0.0092776, 0.0099004, 0.0093329, 0.0099249, -0.0005217, 0.0005533
7: -0.0194269, -0.0158444, -0.0191090, -0.0157033, -0.0027178, 0.0022475
8: 0.9681305, 0.9783950, 0.9690413, 0.9787990, -0.0078597, 0.0066577
9: 0.0040029, 0.0070196, 0.0038841, 0.0067519, -0.0019099, 0.0022953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0049227
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002790, 0.0000268, -0.0002190, 0.0002638
1: -0.0000279, 0.0014503, 0.0000141, 0.0013627, -0.0009900, 0.0010136
2: 0.0141680, 0.0163817, 0.0142991, 0.0163189, -0.0015149, 0.0014721
3: 0.0000268, 0.0016914, 0.0001254, 0.0016442, -0.0011378, 0.0011022
4: -0.0043549, -0.0028195, -0.0042639, -0.0028630, -0.0010602, 0.0010547
5: 0.0079650, 0.0096265, 0.0080634, 0.0095794, -0.0011356, 0.0010997
6: 0.0093001, 0.0099272, 0.0093179, 0.0098900, -0.0005464, 0.0004618
7: -0.0192976, -0.0156905, -0.0191953, -0.0159042, -0.0023407, 0.0024489
8: 0.9685011, 0.9788356, 0.9687942, 0.9782236, -0.0068834, 0.0070755
9: 0.0038734, 0.0069107, 0.0040533, 0.0068246, -0.0020676, 0.0019854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051904, upper bound: 0.0050605
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051879, upper bound: 0.0050605
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002808, 0.0000335, -0.0002276, 0.0002682
1: -0.0000279, 0.0014503, 0.0000058, 0.0013729, -0.0010118, 0.0010347
2: 0.0141680, 0.0163817, 0.0142839, 0.0163313, -0.0015463, 0.0015046
3: 0.0000268, 0.0016914, 0.0001140, 0.0016536, -0.0011613, 0.0011266
4: -0.0043549, -0.0028195, -0.0042745, -0.0028544, -0.0010842, 0.0010773
5: 0.0079650, 0.0096265, 0.0080519, 0.0095887, -0.0011591, 0.0011241
6: 0.0093001, 0.0099272, 0.0093144, 0.0098943, -0.0005556, 0.0004806
7: -0.0192976, -0.0156905, -0.0192155, -0.0158793, -0.0023938, 0.0024999
8: 0.9685011, 0.9788356, 0.9687363, 0.9782947, -0.0070353, 0.0072219
9: 0.0038734, 0.0069107, 0.0040323, 0.0068416, -0.0021104, 0.0020300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051904, upper bound: 0.0050605
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051879, upper bound: 0.0050605
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002790, 0.0000268, -0.0002217, 0.0002696
1: -0.0000382, 0.0014603, 0.0000141, 0.0013627, -0.0010091, 0.0010246
2: 0.0141530, 0.0163971, 0.0142991, 0.0163189, -0.0015313, 0.0014998
3: 0.0000156, 0.0017031, 0.0001254, 0.0016442, -0.0011501, 0.0011227
4: -0.0043653, -0.0028087, -0.0042639, -0.0028630, -0.0010715, 0.0010771
5: 0.0079537, 0.0096381, 0.0080634, 0.0095794, -0.0011479, 0.0011202
6: 0.0092958, 0.0099314, 0.0093179, 0.0098900, -0.0005624, 0.0004665
7: -0.0193227, -0.0156661, -0.0191953, -0.0159042, -0.0023744, 0.0024757
8: 0.9684289, 0.9789057, 0.9687942, 0.9782236, -0.0070134, 0.0071521
9: 0.0038528, 0.0069319, 0.0040533, 0.0068246, -0.0020901, 0.0020175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0051272
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051322
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002808, 0.0000335, -0.0002263, 0.0002710
1: -0.0000382, 0.0014603, 0.0000058, 0.0013729, -0.0010103, 0.0010340
2: 0.0141530, 0.0163971, 0.0142839, 0.0163313, -0.0015445, 0.0015010
3: 0.0000156, 0.0017031, 0.0001140, 0.0016536, -0.0011595, 0.0011234
4: -0.0043653, -0.0028087, -0.0042745, -0.0028544, -0.0010841, 0.0010781
5: 0.0079537, 0.0096381, 0.0080519, 0.0095887, -0.0011572, 0.0011208
6: 0.0092958, 0.0099314, 0.0093144, 0.0098943, -0.0005705, 0.0004839
7: -0.0193227, -0.0156661, -0.0192155, -0.0158793, -0.0023803, 0.0024904
8: 0.9684289, 0.9789057, 0.9687363, 0.9782947, -0.0070200, 0.0072145
9: 0.0038528, 0.0069319, 0.0040323, 0.0068416, -0.0021042, 0.0020201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0051272
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051322
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002694, 0.0000739, -0.0002591, 0.0002494
1: -0.0000279, 0.0014503, 0.0000590, 0.0014349, -0.0010350, 0.0009697
2: 0.0141680, 0.0163817, 0.0141911, 0.0162516, -0.0014407, 0.0015415
3: 0.0000268, 0.0016914, 0.0000442, 0.0015936, -0.0010787, 0.0011554
4: -0.0043549, -0.0028195, -0.0043389, -0.0029096, -0.0010401, 0.0011002
5: 0.0079650, 0.0096265, 0.0079823, 0.0095289, -0.0010763, 0.0011530
6: 0.0093001, 0.0099272, 0.0093370, 0.0099206, -0.0005728, 0.0005548
7: -0.0192976, -0.0156905, -0.0190857, -0.0157282, -0.0024626, 0.0022881
8: 0.9685011, 0.9788356, 0.9691080, 0.9787278, -0.0072048, 0.0067370
9: 0.0038734, 0.0069107, 0.0039050, 0.0067323, -0.0019422, 0.0020870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051471
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002714, 0.0000805, -0.0002667, 0.0002534
1: -0.0000279, 0.0014503, 0.0000494, 0.0014451, -0.0010528, 0.0009929
2: 0.0141680, 0.0163817, 0.0141759, 0.0162660, -0.0014751, 0.0015681
3: 0.0000268, 0.0016914, 0.0000328, 0.0016044, -0.0011039, 0.0011755
4: -0.0043549, -0.0028195, -0.0043494, -0.0028997, -0.0010641, 0.0011187
5: 0.0079650, 0.0096265, 0.0079709, 0.0095397, -0.0011014, 0.0011730
6: 0.0093001, 0.0099272, 0.0093329, 0.0099249, -0.0005803, 0.0005719
7: -0.0192976, -0.0156905, -0.0191090, -0.0157033, -0.0025060, 0.0023337
8: 0.9685011, 0.9788356, 0.9690413, 0.9787990, -0.0073291, 0.0068986
9: 0.0038734, 0.0069107, 0.0038841, 0.0067519, -0.0019833, 0.0021236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051471
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002694, 0.0000739, -0.0002614, 0.0002556
1: -0.0000382, 0.0014603, 0.0000590, 0.0014349, -0.0010561, 0.0009806
2: 0.0141530, 0.0163971, 0.0141911, 0.0162516, -0.0014571, 0.0015700
3: 0.0000156, 0.0017031, 0.0000442, 0.0015936, -0.0010910, 0.0011755
4: -0.0043653, -0.0028087, -0.0043389, -0.0029096, -0.0010515, 0.0011253
5: 0.0079537, 0.0096381, 0.0079823, 0.0095289, -0.0010886, 0.0011729
6: 0.0092958, 0.0099314, 0.0093370, 0.0099206, -0.0005891, 0.0005595
7: -0.0193227, -0.0156661, -0.0190857, -0.0157282, -0.0024912, 0.0023149
8: 0.9684289, 0.9789057, 0.9691080, 0.9787278, -0.0073413, 0.0068136
9: 0.0038528, 0.0069319, 0.0039050, 0.0067323, -0.0019647, 0.0021156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052281
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002714, 0.0000805, -0.0002666, 0.0002570
1: -0.0000382, 0.0014603, 0.0000494, 0.0014451, -0.0010564, 0.0009945
2: 0.0141530, 0.0163971, 0.0141759, 0.0162660, -0.0014755, 0.0015710
3: 0.0000156, 0.0017031, 0.0000328, 0.0016044, -0.0011038, 0.0011772
4: -0.0043653, -0.0028087, -0.0043494, -0.0028997, -0.0010693, 0.0011266
5: 0.0079537, 0.0096381, 0.0079709, 0.0095397, -0.0011013, 0.0011747
6: 0.0092958, 0.0099314, 0.0093329, 0.0099249, -0.0005981, 0.0005818
7: -0.0193227, -0.0156661, -0.0191090, -0.0157033, -0.0025049, 0.0023326
8: 0.9684289, 0.9789057, 0.9690413, 0.9787990, -0.0073454, 0.0069023
9: 0.0038528, 0.0069319, 0.0038841, 0.0067519, -0.0019821, 0.0021240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052281
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002973, 0.0000352, -0.0002099, 0.0002099
1: -0.0000716, 0.0013756, -0.0000716, 0.0013756, -0.0009060, 0.0009060
2: 0.0142798, 0.0164472, 0.0142798, 0.0164472, -0.0013526, 0.0013526
3: 0.0001109, 0.0017407, 0.0001109, 0.0017407, -0.0010151, 0.0010151
4: -0.0042773, -0.0027740, -0.0042773, -0.0027740, -0.0009521, 0.0009521
5: 0.0080489, 0.0096757, 0.0080489, 0.0096757, -0.0010131, 0.0010131
6: 0.0092816, 0.0098955, 0.0092816, 0.0098955, -0.0004340, 0.0004340
7: -0.0194043, -0.0158728, -0.0194043, -0.0158728, -0.0021761, 0.0021761
8: 0.9681953, 0.9783136, 0.9681953, 0.9783136, -0.0063189, 0.0063189
9: 0.0040268, 0.0070006, 0.0040268, 0.0070006, -0.0018401, 0.0018401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051986, upper bound: 0.0049294
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051966, upper bound: 0.0049338
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002880, 0.0000840, -0.0002644, 0.0002090
1: -0.0000716, 0.0013756, -0.0000279, 0.0014503, -0.0010238, 0.0009307
2: 0.0142798, 0.0164472, 0.0141680, 0.0163817, -0.0013795, 0.0015289
3: 0.0001109, 0.0017407, 0.0000268, 0.0016914, -0.0010309, 0.0011477
4: -0.0042773, -0.0027740, -0.0043549, -0.0028195, -0.0010037, 0.0010744
5: 0.0080489, 0.0096757, 0.0079650, 0.0096265, -0.0010285, 0.0011455
6: 0.0092816, 0.0098955, 0.0093001, 0.0099272, -0.0004839, 0.0005485
7: -0.0194043, -0.0158728, -0.0192976, -0.0156905, -0.0024635, 0.0021735
8: 0.9681953, 0.9783136, 0.9685011, 0.9788356, -0.0071424, 0.0064547
9: 0.0040268, 0.0070006, 0.0038734, 0.0069107, -0.0018483, 0.0020821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051986, upper bound: 0.0049294
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051966, upper bound: 0.0049338
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002973, 0.0000352, -0.0002143, 0.0002189
1: -0.0000809, 0.0013873, -0.0000716, 0.0013756, -0.0009271, 0.0009269
2: 0.0142624, 0.0164611, 0.0142798, 0.0164472, -0.0013838, 0.0013833
3: 0.0000978, 0.0017511, 0.0001109, 0.0017407, -0.0010386, 0.0010378
4: -0.0042894, -0.0027644, -0.0042773, -0.0027740, -0.0009737, 0.0009774
5: 0.0080358, 0.0096861, 0.0080489, 0.0096757, -0.0010365, 0.0010357
6: 0.0092776, 0.0099004, 0.0092816, 0.0098955, -0.0004546, 0.0004428
7: -0.0194269, -0.0158444, -0.0194043, -0.0158728, -0.0022256, 0.0022271
8: 0.9681305, 0.9783950, 0.9681953, 0.9783136, -0.0064631, 0.0064649
9: 0.0040029, 0.0070196, 0.0040268, 0.0070006, -0.0018830, 0.0018815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002880, 0.0000840, -0.0002687, 0.0002180
1: -0.0000809, 0.0013873, -0.0000279, 0.0014503, -0.0010449, 0.0009516
2: 0.0142624, 0.0164611, 0.0141680, 0.0163817, -0.0014107, 0.0015596
3: 0.0000978, 0.0017511, 0.0000268, 0.0016914, -0.0010544, 0.0011704
4: -0.0042894, -0.0027644, -0.0043549, -0.0028195, -0.0010254, 0.0010997
5: 0.0080358, 0.0096861, 0.0079650, 0.0096265, -0.0010519, 0.0011681
6: 0.0092776, 0.0099004, 0.0093001, 0.0099272, -0.0005045, 0.0005573
7: -0.0194269, -0.0158444, -0.0192976, -0.0156905, -0.0025130, 0.0022244
8: 0.9681305, 0.9783950, 0.9685011, 0.9788356, -0.0072866, 0.0066007
9: 0.0040029, 0.0070196, 0.0038734, 0.0069107, -0.0018912, 0.0021235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002993, 0.0000428, -0.0002189, 0.0002143
1: -0.0000716, 0.0013756, -0.0000809, 0.0013873, -0.0009269, 0.0009271
2: 0.0142798, 0.0164472, 0.0142624, 0.0164611, -0.0013833, 0.0013838
3: 0.0001109, 0.0017407, 0.0000978, 0.0017511, -0.0010378, 0.0010386
4: -0.0042773, -0.0027740, -0.0042894, -0.0027644, -0.0009774, 0.0009737
5: 0.0080489, 0.0096757, 0.0080358, 0.0096861, -0.0010357, 0.0010365
6: 0.0092816, 0.0098955, 0.0092776, 0.0099004, -0.0004428, 0.0004546
7: -0.0194043, -0.0158728, -0.0194269, -0.0158444, -0.0022271, 0.0022256
8: 0.9681953, 0.9783136, 0.9681305, 0.9783950, -0.0064649, 0.0064631
9: 0.0040268, 0.0070006, 0.0040029, 0.0070196, -0.0018815, 0.0018830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052246, upper bound: 0.0049007
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052249, upper bound: 0.0049109
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002973, 0.0000352, -0.0002902, 0.0000905, -0.0002709, 0.0002115
1: -0.0000716, 0.0013756, -0.0000382, 0.0014603, -0.0010341, 0.0009465
2: 0.0142798, 0.0164472, 0.0141530, 0.0163971, -0.0014039, 0.0015444
3: 0.0001109, 0.0017407, 0.0000156, 0.0017031, -0.0010495, 0.0011594
4: -0.0042773, -0.0027740, -0.0043653, -0.0028087, -0.0010203, 0.0010851
5: 0.0080489, 0.0096757, 0.0079537, 0.0096381, -0.0010470, 0.0011571
6: 0.0092816, 0.0098955, 0.0092958, 0.0099314, -0.0004883, 0.0005648
7: -0.0194043, -0.0158728, -0.0193227, -0.0156661, -0.0024888, 0.0022060
8: 0.9681953, 0.9783136, 0.9684289, 0.9789057, -0.0072146, 0.0065680
9: 0.0040268, 0.0070006, 0.0038528, 0.0069319, -0.0018790, 0.0021034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052246, upper bound: 0.0049007
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052249, upper bound: 0.0049109
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002993, 0.0000428, -0.0002174, 0.0002174
1: -0.0000809, 0.0013873, -0.0000809, 0.0013873, -0.0009248, 0.0009248
2: 0.0142624, 0.0164611, 0.0142624, 0.0164611, -0.0013791, 0.0013791
3: 0.0000978, 0.0017511, 0.0000978, 0.0017511, -0.0010345, 0.0010345
4: -0.0042894, -0.0027644, -0.0042894, -0.0027644, -0.0009754, 0.0009754
5: 0.0080358, 0.0096861, 0.0080358, 0.0096861, -0.0010323, 0.0010323
6: 0.0092776, 0.0099004, 0.0092776, 0.0099004, -0.0004586, 0.0004586
7: -0.0194269, -0.0158444, -0.0194269, -0.0158444, -0.0022117, 0.0022117
8: 0.9681305, 0.9783950, 0.9681305, 0.9783950, -0.0064441, 0.0064441
9: 0.0040029, 0.0070196, 0.0040029, 0.0070196, -0.0018721, 0.0018721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002993, 0.0000428, -0.0002902, 0.0000905, -0.0002710, 0.0002166
1: -0.0000809, 0.0013873, -0.0000382, 0.0014603, -0.0010426, 0.0009512
2: 0.0142624, 0.0164611, 0.0141530, 0.0163971, -0.0014089, 0.0015557
3: 0.0000978, 0.0017511, 0.0000156, 0.0017031, -0.0010528, 0.0011672
4: -0.0042894, -0.0027644, -0.0043653, -0.0028087, -0.0010281, 0.0010979
5: 0.0080358, 0.0096861, 0.0079537, 0.0096381, -0.0010502, 0.0011649
6: 0.0092776, 0.0099004, 0.0092958, 0.0099314, -0.0005086, 0.0005731
7: -0.0194269, -0.0158444, -0.0193227, -0.0156661, -0.0024994, 0.0022103
8: 0.9681305, 0.9783950, 0.9684289, 0.9789057, -0.0072684, 0.0065932
9: 0.0040029, 0.0070196, 0.0038528, 0.0069319, -0.0018819, 0.0021143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002973, 0.0000352, -0.0002090, 0.0002644
1: -0.0000279, 0.0014503, -0.0000716, 0.0013756, -0.0009307, 0.0010238
2: 0.0141680, 0.0163817, 0.0142798, 0.0164472, -0.0015289, 0.0013795
3: 0.0000268, 0.0016914, 0.0001109, 0.0017407, -0.0011477, 0.0010309
4: -0.0043549, -0.0028195, -0.0042773, -0.0027740, -0.0010744, 0.0010037
5: 0.0079650, 0.0096265, 0.0080489, 0.0096757, -0.0011455, 0.0010285
6: 0.0093001, 0.0099272, 0.0092816, 0.0098955, -0.0005485, 0.0004839
7: -0.0192976, -0.0156905, -0.0194043, -0.0158728, -0.0021735, 0.0024635
8: 0.9685011, 0.9788356, 0.9681953, 0.9783136, -0.0064547, 0.0071424
9: 0.0038734, 0.0069107, 0.0040268, 0.0070006, -0.0020821, 0.0018483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051929, upper bound: 0.0050652
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051917, upper bound: 0.0050652
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002993, 0.0000428, -0.0002180, 0.0002687
1: -0.0000279, 0.0014503, -0.0000809, 0.0013873, -0.0009516, 0.0010449
2: 0.0141680, 0.0163817, 0.0142624, 0.0164611, -0.0015596, 0.0014107
3: 0.0000268, 0.0016914, 0.0000978, 0.0017511, -0.0011704, 0.0010544
4: -0.0043549, -0.0028195, -0.0042894, -0.0027644, -0.0010997, 0.0010254
5: 0.0079650, 0.0096265, 0.0080358, 0.0096861, -0.0011681, 0.0010519
6: 0.0093001, 0.0099272, 0.0092776, 0.0099004, -0.0005573, 0.0005045
7: -0.0192976, -0.0156905, -0.0194269, -0.0158444, -0.0022244, 0.0025130
8: 0.9685011, 0.9788356, 0.9681305, 0.9783950, -0.0066007, 0.0072866
9: 0.0038734, 0.0069107, 0.0040029, 0.0070196, -0.0021235, 0.0018912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051929, upper bound: 0.0050652
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051917, upper bound: 0.0050652
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002973, 0.0000352, -0.0002115, 0.0002709
1: -0.0000382, 0.0014603, -0.0000716, 0.0013756, -0.0009465, 0.0010341
2: 0.0141530, 0.0163971, 0.0142798, 0.0164472, -0.0015444, 0.0014039
3: 0.0000156, 0.0017031, 0.0001109, 0.0017407, -0.0011594, 0.0010495
4: -0.0043653, -0.0028087, -0.0042773, -0.0027740, -0.0010851, 0.0010203
5: 0.0079537, 0.0096381, 0.0080489, 0.0096757, -0.0011571, 0.0010470
6: 0.0092958, 0.0099314, 0.0092816, 0.0098955, -0.0005648, 0.0004883
7: -0.0193227, -0.0156661, -0.0194043, -0.0158728, -0.0022060, 0.0024888
8: 0.9684289, 0.9789057, 0.9681953, 0.9783136, -0.0065680, 0.0072146
9: 0.0038528, 0.0069319, 0.0040268, 0.0070006, -0.0021034, 0.0018790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0051309
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0051360
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002993, 0.0000428, -0.0002166, 0.0002710
1: -0.0000382, 0.0014603, -0.0000809, 0.0013873, -0.0009512, 0.0010426
2: 0.0141530, 0.0163971, 0.0142624, 0.0164611, -0.0015557, 0.0014089
3: 0.0000156, 0.0017031, 0.0000978, 0.0017511, -0.0011672, 0.0010528
4: -0.0043653, -0.0028087, -0.0042894, -0.0027644, -0.0010979, 0.0010281
5: 0.0079537, 0.0096381, 0.0080358, 0.0096861, -0.0011649, 0.0010502
6: 0.0092958, 0.0099314, 0.0092776, 0.0099004, -0.0005731, 0.0005086
7: -0.0193227, -0.0156661, -0.0194269, -0.0158444, -0.0022103, 0.0024994
8: 0.9684289, 0.9789057, 0.9681305, 0.9783950, -0.0065932, 0.0072684
9: 0.0038528, 0.0069319, 0.0040029, 0.0070196, -0.0021143, 0.0018819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0051309
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0051360
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002880, 0.0000840, -0.0002484, 0.0002484
1: -0.0000279, 0.0014503, -0.0000279, 0.0014503, -0.0009734, 0.0009734
2: 0.0141680, 0.0163817, 0.0141680, 0.0163817, -0.0014429, 0.0014429
3: 0.0000268, 0.0016914, 0.0000268, 0.0016914, -0.0010790, 0.0010790
4: -0.0043549, -0.0028195, -0.0043549, -0.0028195, -0.0010520, 0.0010520
5: 0.0079650, 0.0096265, 0.0079650, 0.0096265, -0.0010765, 0.0010765
6: 0.0093001, 0.0099272, 0.0093001, 0.0099272, -0.0005815, 0.0005815
7: -0.0192976, -0.0156905, -0.0192976, -0.0156905, -0.0022754, 0.0022754
8: 0.9685011, 0.9788356, 0.9685011, 0.9788356, -0.0067509, 0.0067509
9: 0.0038734, 0.0069107, 0.0038734, 0.0069107, -0.0019351, 0.0019351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051472
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002880, 0.0000840, -0.0002902, 0.0000905, -0.0002561, 0.0002523
1: -0.0000279, 0.0014503, -0.0000382, 0.0014603, -0.0009912, 0.0009941
2: 0.0141680, 0.0163817, 0.0141530, 0.0163971, -0.0014742, 0.0014696
3: 0.0000268, 0.0016914, 0.0000156, 0.0017031, -0.0011022, 0.0010991
4: -0.0043549, -0.0028195, -0.0043653, -0.0028087, -0.0010726, 0.0010706
5: 0.0079650, 0.0096265, 0.0079537, 0.0096381, -0.0010995, 0.0010965
6: 0.0093001, 0.0099272, 0.0092958, 0.0099314, -0.0005891, 0.0005979
7: -0.0192976, -0.0156905, -0.0193227, -0.0156661, -0.0023189, 0.0023199
8: 0.9685011, 0.9788356, 0.9684289, 0.9789057, -0.0068755, 0.0068973
9: 0.0038734, 0.0069107, 0.0038528, 0.0069319, -0.0019748, 0.0019718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051472
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002880, 0.0000840, -0.0002523, 0.0002561
1: -0.0000382, 0.0014603, -0.0000279, 0.0014503, -0.0009941, 0.0009912
2: 0.0141530, 0.0163971, 0.0141680, 0.0163817, -0.0014696, 0.0014742
3: 0.0000156, 0.0017031, 0.0000268, 0.0016914, -0.0010991, 0.0011022
4: -0.0043653, -0.0028087, -0.0043549, -0.0028195, -0.0010706, 0.0010726
5: 0.0079537, 0.0096381, 0.0079650, 0.0096265, -0.0010965, 0.0010995
6: 0.0092958, 0.0099314, 0.0093001, 0.0099272, -0.0005979, 0.0005891
7: -0.0193227, -0.0156661, -0.0192976, -0.0156905, -0.0023199, 0.0023189
8: 0.9684289, 0.9789057, 0.9685011, 0.9788356, -0.0068973, 0.0068755
9: 0.0038528, 0.0069319, 0.0038734, 0.0069107, -0.0019718, 0.0019748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052282
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000905, -0.0002902, 0.0000905, -0.0002557, 0.0002557
1: -0.0000382, 0.0014603, -0.0000382, 0.0014603, -0.0009959, 0.0009959
2: 0.0141530, 0.0163971, 0.0141530, 0.0163971, -0.0014748, 0.0014748
3: 0.0000156, 0.0017031, 0.0000156, 0.0017031, -0.0011017, 0.0011017
4: -0.0043653, -0.0028087, -0.0043653, -0.0028087, -0.0010796, 0.0010796
5: 0.0079537, 0.0096381, 0.0079537, 0.0096381, -0.0010990, 0.0010990
6: 0.0092958, 0.0099314, 0.0092958, 0.0099314, -0.0006068, 0.0006068
7: -0.0193227, -0.0156661, -0.0193227, -0.0156661, -0.0023142, 0.0023142
8: 0.9684289, 0.9789057, 0.9684289, 0.9789057, -0.0069021, 0.0069021
9: 0.0038528, 0.0069319, 0.0038528, 0.0069319, -0.0019704, 0.0019704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052282
time: 0.72 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.04 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049296, upper bound: 0.0048901
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048946, upper bound: 0.0048946
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049296, upper bound: 0.0048901
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048946, upper bound: 0.0048946
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0049227
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049228
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0049227
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049228
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049502, upper bound: 0.0048592
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049228, upper bound: 0.0048668
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049502, upper bound: 0.0048592
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049228, upper bound: 0.0048668
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0049227
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049228
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0049227
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049228
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049192, upper bound: 0.0050613
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048880, upper bound: 0.0050613
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049192, upper bound: 0.0050613
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048880, upper bound: 0.0050613
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0051343
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0051355
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0051343
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0051355
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051452, upper bound: 0.0051497
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051276, upper bound: 0.0051504
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051452, upper bound: 0.0051497
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051276, upper bound: 0.0051504
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051301, upper bound: 0.0052356
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051094, upper bound: 0.0052382
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051301, upper bound: 0.0052356
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051094, upper bound: 0.0052382
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049296, upper bound: 0.0051920
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048946, upper bound: 0.0051920
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049296, upper bound: 0.0051920
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048946, upper bound: 0.0051920
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0052200
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0052200
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0052203
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0052203
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049502, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049228, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049502, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049228, upper bound: 0.0051443
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0052200
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0052200
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0052203
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0052203
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049192, upper bound: 0.0053095
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048880, upper bound: 0.0053095
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049192, upper bound: 0.0053095
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048880, upper bound: 0.0053095
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0054052
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0054055
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0049076, upper bound: 0.0054052
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0054055
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051380, upper bound: 0.0053837
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051185, upper bound: 0.0053842
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051380, upper bound: 0.0053837
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051185, upper bound: 0.0053842
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051213, upper bound: 0.0055035
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0050970, upper bound: 0.0055046
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051213, upper bound: 0.0055035
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0050970, upper bound: 0.0055046
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051949, upper bound: 0.0048901
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0048946
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051949, upper bound: 0.0048901
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0048946
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0049227
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0049227
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0052207, upper bound: 0.0048592
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0052200, upper bound: 0.0048668
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0052207, upper bound: 0.0048592
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0052200, upper bound: 0.0048668
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0049227
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0049227
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051904, upper bound: 0.0050605
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051879, upper bound: 0.0050605
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051904, upper bound: 0.0050605
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051879, upper bound: 0.0050605
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0051272
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051322
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051499, upper bound: 0.0051272
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051322
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051471
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051471
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052281
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052281
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051986, upper bound: 0.0049294
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051966, upper bound: 0.0049338
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051986, upper bound: 0.0049294
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051966, upper bound: 0.0049338
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0052246, upper bound: 0.0049007
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0052249, upper bound: 0.0049109
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0052246, upper bound: 0.0049007
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0052249, upper bound: 0.0049109
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051929, upper bound: 0.0050652
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051917, upper bound: 0.0050652
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051929, upper bound: 0.0050652
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051917, upper bound: 0.0050652
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0051309
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0051360
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0051309
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0051360
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051472
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051472
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052282
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052282

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000214, -0.0002787, 0.0000264, -0.0001880, 0.0001902
1: 0.0000297, 0.0013544, 0.0000155, 0.0013620, -0.0008174, 0.0008459
2: 0.0143116, 0.0162955, 0.0143002, 0.0163167, -0.0012637, 0.0012217
3: 0.0001349, 0.0016266, 0.0001262, 0.0016426, -0.0009489, 0.0009176
4: -0.0042552, -0.0028793, -0.0042632, -0.0028645, -0.0008857, 0.0008546
5: 0.0080728, 0.0095618, 0.0080642, 0.0095778, -0.0009470, 0.0009158
6: 0.0093246, 0.0098865, 0.0093185, 0.0098897, -0.0003690, 0.0003897
7: -0.0191571, -0.0159246, -0.0191917, -0.0159059, -0.0019749, 0.0020399
8: 0.9689037, 0.9781650, 0.9688044, 0.9782186, -0.0057062, 0.0059028
9: 0.0040704, 0.0067924, 0.0040547, 0.0068216, -0.0017230, 0.0016673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0048901
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0048901
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002752, 0.0000233, -0.0002786, 0.0000264, -0.0001874, 0.0001912
1: 0.0000321, 0.0013573, 0.0000161, 0.0013621, -0.0008155, 0.0008454
2: 0.0143072, 0.0162919, 0.0143001, 0.0163159, -0.0012631, 0.0012185
3: 0.0001315, 0.0016240, 0.0001261, 0.0016420, -0.0009484, 0.0009150
4: -0.0042583, -0.0028817, -0.0042633, -0.0028651, -0.0008852, 0.0008533
5: 0.0080695, 0.0095592, 0.0080641, 0.0095771, -0.0009466, 0.0009132
6: 0.0093256, 0.0098877, 0.0093188, 0.0098897, -0.0003719, 0.0003897
7: -0.0191513, -0.0159174, -0.0191903, -0.0159057, -0.0019676, 0.0020390
8: 0.9689201, 0.9781857, 0.9688083, 0.9782192, -0.0056915, 0.0058998
9: 0.0040644, 0.0067876, 0.0040545, 0.0068204, -0.0017223, 0.0016618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0048946
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0048946
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000214, -0.0002691, 0.0000735, -0.0002423, 0.0001892
1: 0.0000297, 0.0013544, 0.0000604, 0.0014342, -0.0009382, 0.0008672
2: 0.0143116, 0.0162955, 0.0141921, 0.0162495, -0.0012866, 0.0014027
3: 0.0001349, 0.0016266, 0.0000450, 0.0015920, -0.0009624, 0.0010536
4: -0.0042552, -0.0028793, -0.0043381, -0.0029112, -0.0009311, 0.0009801
5: 0.0080728, 0.0095618, 0.0079831, 0.0095273, -0.0009602, 0.0010516
6: 0.0093246, 0.0098865, 0.0093376, 0.0099203, -0.0004203, 0.0004973
7: -0.0191571, -0.0159246, -0.0190821, -0.0157298, -0.0022698, 0.0020359
8: 0.9689037, 0.9781650, 0.9691182, 0.9787231, -0.0065510, 0.0060192
9: 0.0040704, 0.0067924, 0.0039064, 0.0067293, -0.0017292, 0.0019156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050833, upper bound: 0.0048901
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050833, upper bound: 0.0048901
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002752, 0.0000233, -0.0002690, 0.0000735, -0.0002416, 0.0001902
1: 0.0000321, 0.0013573, 0.0000610, 0.0014343, -0.0009363, 0.0008670
2: 0.0143072, 0.0162919, 0.0141920, 0.0162486, -0.0012863, 0.0013995
3: 0.0001315, 0.0016240, 0.0000449, 0.0015914, -0.0009621, 0.0010511
4: -0.0042583, -0.0028817, -0.0043382, -0.0029118, -0.0009310, 0.0009788
5: 0.0080695, 0.0095592, 0.0079830, 0.0095266, -0.0009599, 0.0010491
6: 0.0093256, 0.0098877, 0.0093378, 0.0099204, -0.0004232, 0.0004979
7: -0.0191513, -0.0159174, -0.0190807, -0.0157296, -0.0022625, 0.0020346
8: 0.9689201, 0.9781857, 0.9691223, 0.9787237, -0.0065363, 0.0060177
9: 0.0040644, 0.0067876, 0.0039063, 0.0067281, -0.0017284, 0.0019101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050833, upper bound: 0.0048946
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050833, upper bound: 0.0048946
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002775, 0.0000280, -0.0002787, 0.0000264, -0.0001929, 0.0001989
1: 0.0000211, 0.0013645, 0.0000155, 0.0013620, -0.0008408, 0.0008677
2: 0.0142965, 0.0163084, 0.0143002, 0.0163167, -0.0012964, 0.0012566
3: 0.0001234, 0.0016363, 0.0001262, 0.0016426, -0.0009734, 0.0009436
4: -0.0042658, -0.0028703, -0.0042632, -0.0028645, -0.0009084, 0.0008800
5: 0.0080614, 0.0095715, 0.0080642, 0.0095778, -0.0009715, 0.0009418
6: 0.0093209, 0.0098908, 0.0093185, 0.0098897, -0.0003870, 0.0003990
7: -0.0191781, -0.0158998, -0.0191917, -0.0159059, -0.0020306, 0.0020932
8: 0.9688433, 0.9782360, 0.9688044, 0.9782186, -0.0058691, 0.0060553
9: 0.0040496, 0.0068101, 0.0040547, 0.0068216, -0.0017678, 0.0017143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049227
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049227
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002768, 0.0000301, -0.0002786, 0.0000264, -0.0001923, 0.0002002
1: 0.0000242, 0.0013677, 0.0000161, 0.0013621, -0.0008389, 0.0008678
2: 0.0142916, 0.0163038, 0.0143001, 0.0163159, -0.0012967, 0.0012534
3: 0.0001198, 0.0016329, 0.0001261, 0.0016420, -0.0009737, 0.0009411
4: -0.0042691, -0.0028735, -0.0042633, -0.0028651, -0.0009085, 0.0008789
5: 0.0080578, 0.0095681, 0.0080641, 0.0095771, -0.0009718, 0.0009393
6: 0.0093222, 0.0098921, 0.0093188, 0.0098897, -0.0003904, 0.0003992
7: -0.0191707, -0.0158920, -0.0191903, -0.0159057, -0.0020236, 0.0020938
8: 0.9688647, 0.9782584, 0.9688083, 0.9782192, -0.0058545, 0.0060567
9: 0.0040430, 0.0068038, 0.0040545, 0.0068204, -0.0017684, 0.0017089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049228
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049228
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002775, 0.0000280, -0.0002691, 0.0000735, -0.0002471, 0.0001979
1: 0.0000211, 0.0013645, 0.0000604, 0.0014342, -0.0009617, 0.0008891
2: 0.0142965, 0.0163084, 0.0141921, 0.0162495, -0.0013193, 0.0014375
3: 0.0001234, 0.0016363, 0.0000450, 0.0015920, -0.0009870, 0.0010797
4: -0.0042658, -0.0028703, -0.0043381, -0.0029112, -0.0009538, 0.0010055
5: 0.0080614, 0.0095715, 0.0079831, 0.0095273, -0.0009848, 0.0010776
6: 0.0093209, 0.0098908, 0.0093376, 0.0099203, -0.0004383, 0.0005065
7: -0.0191781, -0.0158998, -0.0190821, -0.0157298, -0.0023255, 0.0020891
8: 0.9688433, 0.9782360, 0.9691182, 0.9787231, -0.0067138, 0.0061717
9: 0.0040496, 0.0068101, 0.0039064, 0.0067293, -0.0017741, 0.0019626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049227
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049227
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002768, 0.0000301, -0.0002690, 0.0000735, -0.0002465, 0.0001991
1: 0.0000242, 0.0013677, 0.0000610, 0.0014343, -0.0009597, 0.0008895
2: 0.0142916, 0.0163038, 0.0141920, 0.0162486, -0.0013199, 0.0014343
3: 0.0001198, 0.0016329, 0.0000449, 0.0015914, -0.0009873, 0.0010772
4: -0.0042691, -0.0028735, -0.0043382, -0.0029118, -0.0009543, 0.0010044
5: 0.0080578, 0.0095681, 0.0079830, 0.0095266, -0.0009851, 0.0010751
6: 0.0093222, 0.0098921, 0.0093378, 0.0099204, -0.0004416, 0.0005074
7: -0.0191707, -0.0158920, -0.0190807, -0.0157296, -0.0023185, 0.0020893
8: 0.9688647, 0.9782584, 0.9691223, 0.9787237, -0.0066993, 0.0061746
9: 0.0040430, 0.0068038, 0.0039063, 0.0067281, -0.0017745, 0.0019572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049228
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049228
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000214, -0.0002805, 0.0000330, -0.0001972, 0.0001947
1: 0.0000297, 0.0013544, 0.0000072, 0.0013722, -0.0008435, 0.0008670
2: 0.0143116, 0.0162955, 0.0142849, 0.0163292, -0.0012951, 0.0012609
3: 0.0001349, 0.0016266, 0.0001148, 0.0016519, -0.0009725, 0.0009470
4: -0.0042552, -0.0028793, -0.0042738, -0.0028559, -0.0009098, 0.0008818
5: 0.0080728, 0.0095618, 0.0080527, 0.0095871, -0.0009706, 0.0009452
6: 0.0093246, 0.0098865, 0.0093150, 0.0098940, -0.0003801, 0.0004085
7: -0.0191571, -0.0159246, -0.0192120, -0.0158810, -0.0020387, 0.0020910
8: 0.9689037, 0.9781650, 0.9687462, 0.9782898, -0.0058890, 0.0060495
9: 0.0040704, 0.0067924, 0.0040338, 0.0068386, -0.0017661, 0.0017210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0048592
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0048592
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002752, 0.0000233, -0.0002803, 0.0000331, -0.0001965, 0.0001957
1: 0.0000321, 0.0013573, 0.0000079, 0.0013723, -0.0008416, 0.0008664
2: 0.0143072, 0.0162919, 0.0142848, 0.0163282, -0.0012943, 0.0012576
3: 0.0001315, 0.0016240, 0.0001147, 0.0016512, -0.0009719, 0.0009444
4: -0.0042583, -0.0028817, -0.0042739, -0.0028565, -0.0009092, 0.0008804
5: 0.0080695, 0.0095592, 0.0080526, 0.0095864, -0.0009700, 0.0009426
6: 0.0093256, 0.0098877, 0.0093153, 0.0098941, -0.0003830, 0.0004086
7: -0.0191513, -0.0159174, -0.0192104, -0.0158809, -0.0020314, 0.0020896
8: 0.9689201, 0.9781857, 0.9687508, 0.9782904, -0.0058742, 0.0060456
9: 0.0040644, 0.0067876, 0.0040336, 0.0068373, -0.0017649, 0.0017154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0048668
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0048668
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000214, -0.0002711, 0.0000801, -0.0002488, 0.0001918
1: 0.0000297, 0.0013544, 0.0000509, 0.0014444, -0.0009504, 0.0008842
2: 0.0143116, 0.0162955, 0.0141769, 0.0162638, -0.0013126, 0.0014209
3: 0.0001349, 0.0016266, 0.0000335, 0.0016028, -0.0009820, 0.0010673
4: -0.0042552, -0.0028793, -0.0043487, -0.0029012, -0.0009492, 0.0009928
5: 0.0080728, 0.0095618, 0.0079716, 0.0095380, -0.0009797, 0.0010653
6: 0.0093246, 0.0098865, 0.0093335, 0.0099246, -0.0004254, 0.0005138
7: -0.0191571, -0.0159246, -0.0191054, -0.0157050, -0.0022994, 0.0020693
8: 0.9689037, 0.9781650, 0.9690516, 0.9787943, -0.0066360, 0.0061398
9: 0.0040704, 0.0067924, 0.0038855, 0.0067489, -0.0017611, 0.0019406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051343, upper bound: 0.0048592
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051343, upper bound: 0.0048592
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002752, 0.0000233, -0.0002710, 0.0000801, -0.0002482, 0.0001928
1: 0.0000321, 0.0013573, 0.0000514, 0.0014444, -0.0009484, 0.0008839
2: 0.0143072, 0.0162919, 0.0141768, 0.0162629, -0.0013122, 0.0014176
3: 0.0001315, 0.0016240, 0.0000334, 0.0016021, -0.0009816, 0.0010647
4: -0.0042583, -0.0028817, -0.0043488, -0.0029018, -0.0009491, 0.0009914
5: 0.0080695, 0.0095592, 0.0079715, 0.0095374, -0.0009793, 0.0010627
6: 0.0093256, 0.0098877, 0.0093338, 0.0099247, -0.0004283, 0.0005143
7: -0.0191513, -0.0159174, -0.0191041, -0.0157048, -0.0022921, 0.0020683
8: 0.9689201, 0.9781857, 0.9690555, 0.9787948, -0.0066211, 0.0061381
9: 0.0040644, 0.0067876, 0.0038854, 0.0067478, -0.0017603, 0.0019350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051343, upper bound: 0.0048668
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051343, upper bound: 0.0048668
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002775, 0.0000280, -0.0002805, 0.0000330, -0.0001958, 0.0001977
1: 0.0000211, 0.0013645, 0.0000072, 0.0013722, -0.0008383, 0.0008663
2: 0.0142965, 0.0163084, 0.0142849, 0.0163292, -0.0012934, 0.0012521
3: 0.0001234, 0.0016363, 0.0001148, 0.0016519, -0.0009707, 0.0009401
4: -0.0042658, -0.0028703, -0.0042738, -0.0028559, -0.0009096, 0.0008787
5: 0.0080614, 0.0095715, 0.0080527, 0.0095871, -0.0009688, 0.0009382
6: 0.0093209, 0.0098908, 0.0093150, 0.0098940, -0.0003908, 0.0004117
7: -0.0191781, -0.0158998, -0.0192120, -0.0158810, -0.0020193, 0.0020816
8: 0.9688433, 0.9782360, 0.9687462, 0.9782898, -0.0058489, 0.0060422
9: 0.0040496, 0.0068101, 0.0040338, 0.0068386, -0.0017599, 0.0017063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049227
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049227
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002768, 0.0000301, -0.0002803, 0.0000331, -0.0001952, 0.0001988
1: 0.0000242, 0.0013677, 0.0000079, 0.0013723, -0.0008367, 0.0008658
2: 0.0142916, 0.0163038, 0.0142848, 0.0163282, -0.0012927, 0.0012494
3: 0.0001198, 0.0016329, 0.0001147, 0.0016512, -0.0009702, 0.0009379
4: -0.0042691, -0.0028735, -0.0042739, -0.0028565, -0.0009093, 0.0008781
5: 0.0080578, 0.0095681, 0.0080526, 0.0095864, -0.0009683, 0.0009360
6: 0.0093222, 0.0098921, 0.0093153, 0.0098941, -0.0003947, 0.0004117
7: -0.0191707, -0.0158920, -0.0192104, -0.0158809, -0.0020128, 0.0020807
8: 0.9688647, 0.9782584, 0.9687508, 0.9782904, -0.0058365, 0.0060391
9: 0.0040430, 0.0068038, 0.0040336, 0.0068373, -0.0017591, 0.0017014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049228
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049228
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002775, 0.0000280, -0.0002711, 0.0000801, -0.0002497, 0.0001968
1: 0.0000211, 0.0013645, 0.0000509, 0.0014444, -0.0009592, 0.0008902
2: 0.0142965, 0.0163084, 0.0141769, 0.0162638, -0.0013200, 0.0014332
3: 0.0001234, 0.0016363, 0.0000335, 0.0016028, -0.0009870, 0.0010762
4: -0.0042658, -0.0028703, -0.0043487, -0.0029012, -0.0009579, 0.0010043
5: 0.0080614, 0.0095715, 0.0079716, 0.0095380, -0.0009846, 0.0010741
6: 0.0093209, 0.0098908, 0.0093335, 0.0099246, -0.0004421, 0.0005227
7: -0.0191781, -0.0158998, -0.0191054, -0.0157050, -0.0023143, 0.0020799
8: 0.9688433, 0.9782360, 0.9690516, 0.9787943, -0.0066941, 0.0061752
9: 0.0040496, 0.0068101, 0.0038855, 0.0067489, -0.0017685, 0.0019547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049227
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049227
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002768, 0.0000301, -0.0002710, 0.0000801, -0.0002491, 0.0001980
1: 0.0000242, 0.0013677, 0.0000514, 0.0014444, -0.0009575, 0.0008900
2: 0.0142916, 0.0163038, 0.0141768, 0.0162629, -0.0013196, 0.0014304
3: 0.0001198, 0.0016329, 0.0000334, 0.0016021, -0.0009867, 0.0010740
4: -0.0042691, -0.0028735, -0.0043488, -0.0029018, -0.0009577, 0.0010036
5: 0.0080578, 0.0095681, 0.0079715, 0.0095374, -0.0009844, 0.0010719
6: 0.0093222, 0.0098921, 0.0093338, 0.0099247, -0.0004460, 0.0005232
7: -0.0191707, -0.0158920, -0.0191041, -0.0157048, -0.0023078, 0.0020788
8: 0.9688647, 0.9782584, 0.9690555, 0.9787948, -0.0066816, 0.0061734
9: 0.0040430, 0.0068038, 0.0038854, 0.0067478, -0.0017679, 0.0019498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049228
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049228
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002660, 0.0000687, -0.0002787, 0.0000264, -0.0001872, 0.0002436
1: 0.0000748, 0.0014269, 0.0000155, 0.0013620, -0.0008380, 0.0009629
2: 0.0142031, 0.0162280, 0.0143002, 0.0163167, -0.0014390, 0.0012450
3: 0.0000532, 0.0015759, 0.0001262, 0.0016426, -0.0010807, 0.0009319
4: -0.0043305, -0.0029260, -0.0042632, -0.0028645, -0.0010073, 0.0008982
5: 0.0079913, 0.0095112, 0.0080642, 0.0095778, -0.0010786, 0.0009298
6: 0.0093437, 0.0099172, 0.0093185, 0.0098897, -0.0004738, 0.0004394
7: -0.0190472, -0.0157477, -0.0191917, -0.0159059, -0.0019734, 0.0023255
8: 0.9692186, 0.9786719, 0.9688044, 0.9782186, -0.0058223, 0.0067211
9: 0.0039215, 0.0066999, 0.0040547, 0.0068216, -0.0019635, 0.0016762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0050833
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0050833
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002655, 0.0000713, -0.0002786, 0.0000264, -0.0001866, 0.0002457
1: 0.0000771, 0.0014308, 0.0000161, 0.0013621, -0.0008374, 0.0009656
2: 0.0141971, 0.0162246, 0.0143001, 0.0163159, -0.0014431, 0.0012432
3: 0.0000487, 0.0015733, 0.0001261, 0.0016420, -0.0010838, 0.0009302
4: -0.0043347, -0.0029284, -0.0042633, -0.0028651, -0.0010100, 0.0008990
5: 0.0079868, 0.0095086, 0.0080641, 0.0095771, -0.0010817, 0.0009281
6: 0.0093446, 0.0099189, 0.0093188, 0.0098897, -0.0004799, 0.0004407
7: -0.0190416, -0.0157380, -0.0191903, -0.0159057, -0.0019668, 0.0023323
8: 0.9692345, 0.9786997, 0.9688083, 0.9782192, -0.0058146, 0.0067402
9: 0.0039133, 0.0066951, 0.0040545, 0.0068204, -0.0019693, 0.0016716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0050833
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0050833
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002660, 0.0000687, -0.0002805, 0.0000330, -0.0001964, 0.0002481
1: 0.0000748, 0.0014269, 0.0000072, 0.0013722, -0.0008642, 0.0009840
2: 0.0142031, 0.0162280, 0.0142849, 0.0163292, -0.0014704, 0.0012841
3: 0.0000532, 0.0015759, 0.0001148, 0.0016519, -0.0011043, 0.0009613
4: -0.0043305, -0.0029260, -0.0042738, -0.0028559, -0.0010314, 0.0009253
5: 0.0079913, 0.0095112, 0.0080527, 0.0095871, -0.0011022, 0.0009592
6: 0.0093437, 0.0099172, 0.0093150, 0.0098940, -0.0004849, 0.0004581
7: -0.0190472, -0.0157477, -0.0192120, -0.0158810, -0.0020372, 0.0023766
8: 0.9692186, 0.9786719, 0.9687462, 0.9782898, -0.0060050, 0.0068677
9: 0.0039215, 0.0066999, 0.0040338, 0.0068386, -0.0020065, 0.0017299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0050613
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0050613
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002655, 0.0000713, -0.0002803, 0.0000331, -0.0001958, 0.0002501
1: 0.0000771, 0.0014308, 0.0000079, 0.0013723, -0.0008635, 0.0009866
2: 0.0141971, 0.0162246, 0.0142848, 0.0163282, -0.0014743, 0.0012823
3: 0.0000487, 0.0015733, 0.0001147, 0.0016512, -0.0011072, 0.0009596
4: -0.0043347, -0.0029284, -0.0042739, -0.0028565, -0.0010340, 0.0009262
5: 0.0079868, 0.0095086, 0.0080526, 0.0095864, -0.0011051, 0.0009575
6: 0.0093446, 0.0099189, 0.0093153, 0.0098941, -0.0004909, 0.0004596
7: -0.0190416, -0.0157380, -0.0192104, -0.0158809, -0.0020306, 0.0023829
8: 0.9692345, 0.9786997, 0.9687508, 0.9782904, -0.0059973, 0.0068859
9: 0.0039133, 0.0066951, 0.0040336, 0.0068373, -0.0020119, 0.0017253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0050613
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0050613
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0000753, -0.0002787, 0.0000264, -0.0001904, 0.0002501
1: 0.0000653, 0.0014370, 0.0000155, 0.0013620, -0.0008569, 0.0009737
2: 0.0141879, 0.0162421, 0.0143002, 0.0163167, -0.0014552, 0.0012735
3: 0.0000418, 0.0015865, 0.0001262, 0.0016426, -0.0010929, 0.0009531
4: -0.0043411, -0.0029163, -0.0042632, -0.0028645, -0.0010186, 0.0009174
5: 0.0079799, 0.0095218, 0.0080642, 0.0095778, -0.0010908, 0.0009510
6: 0.0093397, 0.0099215, 0.0093185, 0.0098897, -0.0004903, 0.0004440
7: -0.0190702, -0.0157230, -0.0191917, -0.0159059, -0.0020126, 0.0023520
8: 0.9691527, 0.9787427, 0.9688044, 0.9782186, -0.0059555, 0.0067970
9: 0.0039007, 0.0067192, 0.0040547, 0.0068216, -0.0019858, 0.0017119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051343
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051343
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002676, 0.0000775, -0.0002786, 0.0000264, -0.0001898, 0.0002516
1: 0.0000674, 0.0014404, 0.0000161, 0.0013621, -0.0008558, 0.0009760
2: 0.0141828, 0.0162391, 0.0143001, 0.0163159, -0.0014587, 0.0012714
3: 0.0000380, 0.0015842, 0.0001261, 0.0016420, -0.0010956, 0.0009513
4: -0.0043446, -0.0029184, -0.0042633, -0.0028651, -0.0010209, 0.0009178
5: 0.0079761, 0.0095195, 0.0080641, 0.0095771, -0.0010935, 0.0009492
6: 0.0093405, 0.0099230, 0.0093188, 0.0098897, -0.0004956, 0.0004451
7: -0.0190652, -0.0157146, -0.0191903, -0.0159057, -0.0020061, 0.0023578
8: 0.9691668, 0.9787666, 0.9688083, 0.9782192, -0.0059463, 0.0068132
9: 0.0038936, 0.0067151, 0.0040545, 0.0068204, -0.0019907, 0.0017073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051355
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051355
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0000753, -0.0002805, 0.0000330, -0.0001951, 0.0002508
1: 0.0000653, 0.0014370, 0.0000072, 0.0013722, -0.0008616, 0.0009834
2: 0.0141879, 0.0162421, 0.0142849, 0.0163292, -0.0014688, 0.0012792
3: 0.0000418, 0.0015865, 0.0001148, 0.0016519, -0.0011026, 0.0009569
4: -0.0043411, -0.0029163, -0.0042738, -0.0028559, -0.0010313, 0.0009251
5: 0.0079799, 0.0095218, 0.0080527, 0.0095871, -0.0011005, 0.0009547
6: 0.0093397, 0.0099215, 0.0093150, 0.0098940, -0.0004987, 0.0004614
7: -0.0190702, -0.0157230, -0.0192120, -0.0158810, -0.0020197, 0.0023674
8: 0.9691527, 0.9787427, 0.9687462, 0.9782898, -0.0059834, 0.0068611
9: 0.0039007, 0.0067192, 0.0040338, 0.0068386, -0.0020006, 0.0017173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051343
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051343
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002676, 0.0000775, -0.0002803, 0.0000331, -0.0001945, 0.0002525
1: 0.0000674, 0.0014404, 0.0000079, 0.0013723, -0.0008609, 0.0009858
2: 0.0141828, 0.0162391, 0.0142848, 0.0163282, -0.0014724, 0.0012776
3: 0.0000380, 0.0015842, 0.0001147, 0.0016512, -0.0011053, 0.0009554
4: -0.0043446, -0.0029184, -0.0042739, -0.0028565, -0.0010339, 0.0009260
5: 0.0079761, 0.0095195, 0.0080526, 0.0095864, -0.0011032, 0.0009532
6: 0.0093405, 0.0099230, 0.0093153, 0.0098941, -0.0005041, 0.0004626
7: -0.0190652, -0.0157146, -0.0192104, -0.0158809, -0.0020138, 0.0023734
8: 0.9691668, 0.9787666, 0.9687508, 0.9782904, -0.0059761, 0.0068779
9: 0.0038936, 0.0067151, 0.0040336, 0.0068373, -0.0020057, 0.0017131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051355
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051355
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002660, 0.0000687, -0.0002691, 0.0000735, -0.0002250, 0.0002265
1: 0.0000748, 0.0014269, 0.0000604, 0.0014342, -0.0008763, 0.0009020
2: 0.0142031, 0.0162280, 0.0141921, 0.0162495, -0.0013395, 0.0013029
3: 0.0000532, 0.0015759, 0.0000450, 0.0015920, -0.0010027, 0.0009756
4: -0.0043305, -0.0029260, -0.0043381, -0.0029112, -0.0009695, 0.0009401
5: 0.0079913, 0.0095112, 0.0079831, 0.0095273, -0.0010004, 0.0009734
6: 0.0093437, 0.0099172, 0.0093376, 0.0099203, -0.0005021, 0.0005248
7: -0.0190472, -0.0157477, -0.0190821, -0.0157298, -0.0020690, 0.0021238
8: 0.9692186, 0.9786719, 0.9691182, 0.9787231, -0.0060920, 0.0062644
9: 0.0039215, 0.0066999, 0.0039064, 0.0067293, -0.0018038, 0.0017565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051271, upper bound: 0.0051602
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051271, upper bound: 0.0051602
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002655, 0.0000713, -0.0002690, 0.0000735, -0.0002246, 0.0002281
1: 0.0000771, 0.0014308, 0.0000610, 0.0014343, -0.0008767, 0.0009030
2: 0.0141971, 0.0162246, 0.0141920, 0.0162486, -0.0013410, 0.0013026
3: 0.0000487, 0.0015733, 0.0000449, 0.0015914, -0.0010038, 0.0009751
4: -0.0043347, -0.0029284, -0.0043382, -0.0029118, -0.0009706, 0.0009429
5: 0.0079868, 0.0095086, 0.0079830, 0.0095266, -0.0010015, 0.0009729
6: 0.0093446, 0.0099189, 0.0093378, 0.0099204, -0.0005086, 0.0005259
7: -0.0190416, -0.0157380, -0.0190807, -0.0157296, -0.0020649, 0.0021262
8: 0.9692345, 0.9786997, 0.9691223, 0.9787237, -0.0060917, 0.0062714
9: 0.0039133, 0.0066951, 0.0039063, 0.0067281, -0.0018054, 0.0017541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051271, upper bound: 0.0051610
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051271, upper bound: 0.0051610
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002660, 0.0000687, -0.0002711, 0.0000801, -0.0002330, 0.0002305
1: 0.0000748, 0.0014269, 0.0000509, 0.0014444, -0.0008965, 0.0009253
2: 0.0142031, 0.0162280, 0.0141769, 0.0162638, -0.0013739, 0.0013331
3: 0.0000532, 0.0015759, 0.0000335, 0.0016028, -0.0010278, 0.0009983
4: -0.0043305, -0.0029260, -0.0043487, -0.0029012, -0.0009935, 0.0009611
5: 0.0079913, 0.0095112, 0.0079716, 0.0095380, -0.0010254, 0.0009961
6: 0.0093437, 0.0099172, 0.0093335, 0.0099246, -0.0005106, 0.0005419
7: -0.0190472, -0.0157477, -0.0191054, -0.0157050, -0.0021182, 0.0021694
8: 0.9692186, 0.9786719, 0.9690516, 0.9787943, -0.0062329, 0.0064259
9: 0.0039215, 0.0066999, 0.0038855, 0.0067489, -0.0018448, 0.0017980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051923, upper bound: 0.0051497
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051923, upper bound: 0.0051497
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002655, 0.0000713, -0.0002710, 0.0000801, -0.0002326, 0.0002321
1: 0.0000771, 0.0014308, 0.0000514, 0.0014444, -0.0008968, 0.0009262
2: 0.0141971, 0.0162246, 0.0141768, 0.0162629, -0.0013753, 0.0013328
3: 0.0000487, 0.0015733, 0.0000334, 0.0016021, -0.0010289, 0.0009978
4: -0.0043347, -0.0029284, -0.0043488, -0.0029018, -0.0009946, 0.0009639
5: 0.0079868, 0.0095086, 0.0079715, 0.0095374, -0.0010265, 0.0009955
6: 0.0093446, 0.0099189, 0.0093338, 0.0099247, -0.0005172, 0.0005429
7: -0.0190416, -0.0157380, -0.0191041, -0.0157048, -0.0021140, 0.0021716
8: 0.9692345, 0.9786997, 0.9690555, 0.9787948, -0.0062325, 0.0064328
9: 0.0039133, 0.0066951, 0.0038854, 0.0067478, -0.0018466, 0.0017955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051923, upper bound: 0.0051504
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051923, upper bound: 0.0051504
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0000753, -0.0002691, 0.0000735, -0.0002293, 0.0002340
1: 0.0000653, 0.0014370, 0.0000604, 0.0014342, -0.0009005, 0.0009199
2: 0.0141879, 0.0162421, 0.0141921, 0.0162495, -0.0013663, 0.0013378
3: 0.0000418, 0.0015865, 0.0000450, 0.0015920, -0.0010228, 0.0010011
4: -0.0043411, -0.0029163, -0.0043381, -0.0029112, -0.0009881, 0.0009638
5: 0.0079799, 0.0095218, 0.0079831, 0.0095273, -0.0010205, 0.0009988
6: 0.0093397, 0.0099215, 0.0093376, 0.0099203, -0.0005200, 0.0005324
7: -0.0190702, -0.0157230, -0.0190821, -0.0157298, -0.0021190, 0.0021674
8: 0.9691527, 0.9787427, 0.9691182, 0.9787231, -0.0062567, 0.0063894
9: 0.0039007, 0.0067192, 0.0039064, 0.0067293, -0.0018405, 0.0017989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050968, upper bound: 0.0052356
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050968, upper bound: 0.0052356
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002676, 0.0000775, -0.0002690, 0.0000735, -0.0002290, 0.0002354
1: 0.0000674, 0.0014404, 0.0000610, 0.0014343, -0.0009007, 0.0009216
2: 0.0141828, 0.0162391, 0.0141920, 0.0162486, -0.0013688, 0.0013375
3: 0.0000380, 0.0015842, 0.0000449, 0.0015914, -0.0010247, 0.0010006
4: -0.0043446, -0.0029184, -0.0043382, -0.0029118, -0.0009899, 0.0009657
5: 0.0079761, 0.0095195, 0.0079830, 0.0095266, -0.0010224, 0.0009983
6: 0.0093405, 0.0099230, 0.0093378, 0.0099204, -0.0005253, 0.0005338
7: -0.0190652, -0.0157146, -0.0190807, -0.0157296, -0.0021150, 0.0021715
8: 0.9691668, 0.9787666, 0.9691223, 0.9787237, -0.0062557, 0.0064013
9: 0.0038936, 0.0067151, 0.0039063, 0.0067281, -0.0018436, 0.0017964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050968, upper bound: 0.0052382
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050968, upper bound: 0.0052382
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0000753, -0.0002711, 0.0000801, -0.0002325, 0.0002341
1: 0.0000653, 0.0014370, 0.0000509, 0.0014444, -0.0009017, 0.0009269
2: 0.0141879, 0.0162421, 0.0141769, 0.0162638, -0.0013744, 0.0013372
3: 0.0000418, 0.0015865, 0.0000335, 0.0016028, -0.0010279, 0.0010000
4: -0.0043411, -0.0029163, -0.0043487, -0.0029012, -0.0009988, 0.0009706
5: 0.0079799, 0.0095218, 0.0079716, 0.0095380, -0.0010255, 0.0009977
6: 0.0093397, 0.0099215, 0.0093335, 0.0099246, -0.0005290, 0.0005519
7: -0.0190702, -0.0157230, -0.0191054, -0.0157050, -0.0021127, 0.0021687
8: 0.9691527, 0.9787427, 0.9690516, 0.9787943, -0.0062554, 0.0064300
9: 0.0039007, 0.0067192, 0.0038855, 0.0067489, -0.0018439, 0.0017956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051011, upper bound: 0.0052356
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051011, upper bound: 0.0052356
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002676, 0.0000775, -0.0002710, 0.0000801, -0.0002320, 0.0002356
1: 0.0000674, 0.0014404, 0.0000514, 0.0014444, -0.0009017, 0.0009278
2: 0.0141828, 0.0162391, 0.0141768, 0.0162629, -0.0013757, 0.0013362
3: 0.0000380, 0.0015842, 0.0000334, 0.0016021, -0.0010287, 0.0009990
4: -0.0043446, -0.0029184, -0.0043488, -0.0029018, -0.0009995, 0.0009725
5: 0.0079761, 0.0095195, 0.0079715, 0.0095374, -0.0010263, 0.0009966
6: 0.0093405, 0.0099230, 0.0093338, 0.0099247, -0.0005342, 0.0005527
7: -0.0190652, -0.0157146, -0.0191041, -0.0157048, -0.0021072, 0.0021702
8: 0.9691668, 0.9787666, 0.9690555, 0.9787948, -0.0062519, 0.0064360
9: 0.0038936, 0.0067151, 0.0038854, 0.0067478, -0.0018452, 0.0017918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051011, upper bound: 0.0052382
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051011, upper bound: 0.0052382
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000214, -0.0002970, 0.0000348, -0.0002059, 0.0002183
1: 0.0000297, 0.0013544, -0.0000701, 0.0013750, -0.0008802, 0.0009771
2: 0.0143116, 0.0162955, 0.0142809, 0.0164450, -0.0014604, 0.0013159
3: 0.0001349, 0.0016266, 0.0001117, 0.0017390, -0.0010968, 0.0009884
4: -0.0042552, -0.0028793, -0.0042766, -0.0027756, -0.0010216, 0.0009199
5: 0.0080728, 0.0095618, 0.0080497, 0.0096740, -0.0010947, 0.0009864
6: 0.0093246, 0.0098865, 0.0092822, 0.0098952, -0.0003957, 0.0004446
7: -0.0191571, -0.0159246, -0.0194007, -0.0158744, -0.0021283, 0.0023607
8: 0.9689037, 0.9781650, 0.9682056, 0.9783089, -0.0061456, 0.0068208
9: 0.0040704, 0.0067924, 0.0040282, 0.0069975, -0.0019932, 0.0017964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0051919
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0051920
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002752, 0.0000233, -0.0002969, 0.0000348, -0.0002052, 0.0002197
1: 0.0000321, 0.0013573, -0.0000696, 0.0013749, -0.0008782, 0.0009787
2: 0.0143072, 0.0162919, 0.0142809, 0.0164442, -0.0014629, 0.0013125
3: 0.0001315, 0.0016240, 0.0001117, 0.0017385, -0.0010987, 0.0009857
4: -0.0042583, -0.0028817, -0.0042766, -0.0027761, -0.0010232, 0.0009184
5: 0.0080695, 0.0095592, 0.0080497, 0.0096735, -0.0010965, 0.0009837
6: 0.0093256, 0.0098877, 0.0092824, 0.0098952, -0.0003985, 0.0004454
7: -0.0191513, -0.0159174, -0.0193995, -0.0158744, -0.0021207, 0.0023645
8: 0.9689201, 0.9781857, 0.9682090, 0.9783087, -0.0061301, 0.0068323
9: 0.0040644, 0.0067876, 0.0040282, 0.0069965, -0.0019964, 0.0017907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0051919
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0051920
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000214, -0.0002876, 0.0000835, -0.0002566, 0.0002132
1: 0.0000297, 0.0013544, -0.0000264, 0.0014496, -0.0009763, 0.0009768
2: 0.0143116, 0.0162955, 0.0141690, 0.0163795, -0.0014524, 0.0014598
3: 0.0001349, 0.0016266, 0.0000276, 0.0016898, -0.0010874, 0.0010966
4: -0.0042552, -0.0028793, -0.0043542, -0.0028210, -0.0010407, 0.0010197
5: 0.0080728, 0.0095618, 0.0079657, 0.0096249, -0.0010850, 0.0010944
6: 0.0093246, 0.0098865, 0.0093008, 0.0099269, -0.0004364, 0.0005399
7: -0.0191571, -0.0159246, -0.0192940, -0.0156922, -0.0023628, 0.0023096
8: 0.9689037, 0.9781650, 0.9685115, 0.9788310, -0.0068174, 0.0067914
9: 0.0040704, 0.0067924, 0.0038747, 0.0069077, -0.0019590, 0.0019939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050783, upper bound: 0.0051919
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050783, upper bound: 0.0051920
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002752, 0.0000233, -0.0002875, 0.0000835, -0.0002559, 0.0002145
1: 0.0000321, 0.0013573, -0.0000258, 0.0014496, -0.0009743, 0.0009784
2: 0.0143072, 0.0162919, 0.0141690, 0.0163787, -0.0014547, 0.0014564
3: 0.0001315, 0.0016240, 0.0000276, 0.0016892, -0.0010891, 0.0010939
4: -0.0042583, -0.0028817, -0.0043542, -0.0028215, -0.0010423, 0.0010183
5: 0.0080695, 0.0095592, 0.0079657, 0.0096243, -0.0010867, 0.0010918
6: 0.0093256, 0.0098877, 0.0093010, 0.0099269, -0.0004393, 0.0005403
7: -0.0191513, -0.0159174, -0.0192927, -0.0156922, -0.0023552, 0.0023126
8: 0.9689201, 0.9781857, 0.9685151, 0.9788309, -0.0068020, 0.0068023
9: 0.0040644, 0.0067876, 0.0038748, 0.0069066, -0.0019616, 0.0019881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050783, upper bound: 0.0051919
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050783, upper bound: 0.0051920
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002775, 0.0000280, -0.0002970, 0.0000348, -0.0002108, 0.0002270
1: 0.0000211, 0.0013645, -0.0000701, 0.0013750, -0.0009037, 0.0009989
2: 0.0142965, 0.0163084, 0.0142809, 0.0164450, -0.0014931, 0.0013507
3: 0.0001234, 0.0016363, 0.0001117, 0.0017390, -0.0011214, 0.0010144
4: -0.0042658, -0.0028703, -0.0042766, -0.0027756, -0.0010442, 0.0009453
5: 0.0080614, 0.0095715, 0.0080497, 0.0096740, -0.0011192, 0.0010124
6: 0.0093209, 0.0098908, 0.0092822, 0.0098952, -0.0004137, 0.0004539
7: -0.0191781, -0.0158998, -0.0194007, -0.0158744, -0.0021840, 0.0024139
8: 0.9688433, 0.9782360, 0.9682056, 0.9783089, -0.0063085, 0.0069733
9: 0.0040496, 0.0068101, 0.0040282, 0.0069975, -0.0020380, 0.0018434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052190
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052200
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002768, 0.0000301, -0.0002969, 0.0000348, -0.0002101, 0.0002286
1: 0.0000242, 0.0013677, -0.0000696, 0.0013749, -0.0009016, 0.0010011
2: 0.0142916, 0.0163038, 0.0142809, 0.0164442, -0.0014965, 0.0013473
3: 0.0001198, 0.0016329, 0.0001117, 0.0017385, -0.0011239, 0.0010118
4: -0.0042691, -0.0028735, -0.0042766, -0.0027761, -0.0010466, 0.0009441
5: 0.0080578, 0.0095681, 0.0080497, 0.0096735, -0.0011218, 0.0010098
6: 0.0093222, 0.0098921, 0.0092824, 0.0098952, -0.0004170, 0.0004549
7: -0.0191707, -0.0158920, -0.0193995, -0.0158744, -0.0021767, 0.0024192
8: 0.9688647, 0.9782584, 0.9682090, 0.9783087, -0.0062931, 0.0069892
9: 0.0040430, 0.0068038, 0.0040282, 0.0069965, -0.0020425, 0.0018378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052190
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052200
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002775, 0.0000280, -0.0002876, 0.0000835, -0.0002615, 0.0002218
1: 0.0000211, 0.0013645, -0.0000264, 0.0014496, -0.0009998, 0.0009986
2: 0.0142965, 0.0163084, 0.0141690, 0.0163795, -0.0014851, 0.0014946
3: 0.0001234, 0.0016363, 0.0000276, 0.0016898, -0.0011120, 0.0011226
4: -0.0042658, -0.0028703, -0.0043542, -0.0028210, -0.0010634, 0.0010451
5: 0.0080614, 0.0095715, 0.0079657, 0.0096249, -0.0011095, 0.0011204
6: 0.0093209, 0.0098908, 0.0093008, 0.0099269, -0.0004544, 0.0005492
7: -0.0191781, -0.0158998, -0.0192940, -0.0156922, -0.0024184, 0.0023628
8: 0.9688433, 0.9782360, 0.9685115, 0.9788310, -0.0069802, 0.0069439
9: 0.0040496, 0.0068101, 0.0038747, 0.0069077, -0.0020038, 0.0020409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052190
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052203
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002768, 0.0000301, -0.0002875, 0.0000835, -0.0002608, 0.0002234
1: 0.0000242, 0.0013677, -0.0000258, 0.0014496, -0.0009977, 0.0010009
2: 0.0142916, 0.0163038, 0.0141690, 0.0163787, -0.0014883, 0.0014913
3: 0.0001198, 0.0016329, 0.0000276, 0.0016892, -0.0011144, 0.0011200
4: -0.0042691, -0.0028735, -0.0043542, -0.0028215, -0.0010657, 0.0010439
5: 0.0080578, 0.0095681, 0.0079657, 0.0096243, -0.0011119, 0.0011178
6: 0.0093222, 0.0098921, 0.0093010, 0.0099269, -0.0004577, 0.0005498
7: -0.0191707, -0.0158920, -0.0192927, -0.0156922, -0.0024112, 0.0023673
8: 0.9688647, 0.9782584, 0.9685151, 0.9788309, -0.0069650, 0.0069592
9: 0.0040430, 0.0068038, 0.0038748, 0.0069066, -0.0020077, 0.0020353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052190
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052203
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000214, -0.0002990, 0.0000424, -0.0002131, 0.0002204
1: 0.0000297, 0.0013544, -0.0000793, 0.0013866, -0.0008918, 0.0009912
2: 0.0143116, 0.0162955, 0.0142634, 0.0164588, -0.0014800, 0.0013331
3: 0.0001349, 0.0016266, 0.0000986, 0.0017494, -0.0011110, 0.0010013
4: -0.0042552, -0.0028793, -0.0042887, -0.0027660, -0.0010400, 0.0009319
5: 0.0080728, 0.0095618, 0.0080366, 0.0096844, -0.0011088, 0.0009994
6: 0.0093246, 0.0098865, 0.0092783, 0.0099001, -0.0004006, 0.0004630
7: -0.0191571, -0.0159246, -0.0194232, -0.0158460, -0.0021564, 0.0023857
8: 0.9689037, 0.9781650, 0.9681412, 0.9783903, -0.0062263, 0.0069141
9: 0.0040704, 0.0067924, 0.0040043, 0.0070165, -0.0020156, 0.0018201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0051443
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0051443
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002752, 0.0000233, -0.0002989, 0.0000423, -0.0002123, 0.0002217
1: 0.0000321, 0.0013573, -0.0000789, 0.0013865, -0.0008897, 0.0009925
2: 0.0143072, 0.0162919, 0.0142635, 0.0164582, -0.0014819, 0.0013297
3: 0.0001315, 0.0016240, 0.0000986, 0.0017489, -0.0011124, 0.0009986
4: -0.0042583, -0.0028817, -0.0042886, -0.0027664, -0.0010415, 0.0009304
5: 0.0080695, 0.0095592, 0.0080366, 0.0096839, -0.0011102, 0.0009967
6: 0.0093256, 0.0098877, 0.0092785, 0.0099001, -0.0004034, 0.0004635
7: -0.0191513, -0.0159174, -0.0194222, -0.0158461, -0.0021488, 0.0023886
8: 0.9689201, 0.9781857, 0.9681441, 0.9783898, -0.0062106, 0.0069227
9: 0.0040644, 0.0067876, 0.0040044, 0.0070156, -0.0020181, 0.0018143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0051443
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0051443
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000214, -0.0002898, 0.0000901, -0.0002627, 0.0002159
1: 0.0000297, 0.0013544, -0.0000367, 0.0014597, -0.0009870, 0.0009962
2: 0.0143116, 0.0162955, 0.0141540, 0.0163950, -0.0014804, 0.0014757
3: 0.0001349, 0.0016266, 0.0000163, 0.0017014, -0.0011081, 0.0011086
4: -0.0042552, -0.0028793, -0.0043646, -0.0028102, -0.0010634, 0.0010308
5: 0.0080728, 0.0095618, 0.0079544, 0.0096365, -0.0011056, 0.0011064
6: 0.0093246, 0.0098865, 0.0092964, 0.0099311, -0.0004409, 0.0005560
7: -0.0191571, -0.0159246, -0.0193192, -0.0156676, -0.0023888, 0.0023434
8: 0.9689037, 0.9781650, 0.9684391, 0.9789013, -0.0068920, 0.0069232
9: 0.0040704, 0.0067924, 0.0038541, 0.0069289, -0.0019912, 0.0020158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051272, upper bound: 0.0051443
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051272, upper bound: 0.0051443
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002752, 0.0000233, -0.0002897, 0.0000900, -0.0002619, 0.0002172
1: 0.0000321, 0.0013573, -0.0000361, 0.0014596, -0.0009849, 0.0009971
2: 0.0143072, 0.0162919, 0.0141541, 0.0163941, -0.0014820, 0.0014723
3: 0.0001315, 0.0016240, 0.0000164, 0.0017008, -0.0011094, 0.0011058
4: -0.0042583, -0.0028817, -0.0043645, -0.0028109, -0.0010646, 0.0010293
5: 0.0080695, 0.0095592, 0.0079545, 0.0096358, -0.0011069, 0.0011037
6: 0.0093256, 0.0098877, 0.0092966, 0.0099311, -0.0004438, 0.0005563
7: -0.0191513, -0.0159174, -0.0193178, -0.0156678, -0.0023811, 0.0023461
8: 0.9689201, 0.9781857, 0.9684432, 0.9789007, -0.0068762, 0.0069302
9: 0.0040644, 0.0067876, 0.0038542, 0.0069277, -0.0019936, 0.0020099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051272, upper bound: 0.0051443
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051272, upper bound: 0.0051443
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002775, 0.0000280, -0.0002990, 0.0000424, -0.0002140, 0.0002258
1: 0.0000211, 0.0013645, -0.0000793, 0.0013866, -0.0009006, 0.0009964
2: 0.0142965, 0.0163084, 0.0142634, 0.0164588, -0.0014886, 0.0013455
3: 0.0001234, 0.0016363, 0.0000986, 0.0017494, -0.0011177, 0.0010103
4: -0.0042658, -0.0028703, -0.0042887, -0.0027660, -0.0010435, 0.0009435
5: 0.0080614, 0.0095715, 0.0080366, 0.0096844, -0.0011155, 0.0010084
6: 0.0093209, 0.0098908, 0.0092783, 0.0099001, -0.0004172, 0.0004660
7: -0.0191781, -0.0158998, -0.0194232, -0.0158460, -0.0021716, 0.0024019
8: 0.9688433, 0.9782360, 0.9681412, 0.9783903, -0.0062851, 0.0069534
9: 0.0040496, 0.0068101, 0.0040043, 0.0070165, -0.0020293, 0.0018344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052190
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052200
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002768, 0.0000301, -0.0002989, 0.0000423, -0.0002134, 0.0002273
1: 0.0000242, 0.0013677, -0.0000789, 0.0013865, -0.0008989, 0.0009980
2: 0.0142916, 0.0163038, 0.0142635, 0.0164582, -0.0014910, 0.0013426
3: 0.0001198, 0.0016329, 0.0000986, 0.0017489, -0.0011195, 0.0010080
4: -0.0042691, -0.0028735, -0.0042886, -0.0027664, -0.0010452, 0.0009427
5: 0.0080578, 0.0095681, 0.0080366, 0.0096839, -0.0011173, 0.0010060
6: 0.0093222, 0.0098921, 0.0092785, 0.0099001, -0.0004211, 0.0004665
7: -0.0191707, -0.0158920, -0.0194222, -0.0158461, -0.0021647, 0.0024056
8: 0.9688647, 0.9782584, 0.9681441, 0.9783898, -0.0062717, 0.0069644
9: 0.0040430, 0.0068038, 0.0040044, 0.0070156, -0.0020323, 0.0018293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052190
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052200
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002775, 0.0000280, -0.0002898, 0.0000901, -0.0002640, 0.0002205
1: 0.0000211, 0.0013645, -0.0000367, 0.0014597, -0.0009967, 0.0009969
2: 0.0142965, 0.0163084, 0.0141540, 0.0163950, -0.0014811, 0.0014894
3: 0.0001234, 0.0016363, 0.0000163, 0.0017014, -0.0011085, 0.0011185
4: -0.0042658, -0.0028703, -0.0043646, -0.0028102, -0.0010640, 0.0010433
5: 0.0080614, 0.0095715, 0.0079544, 0.0096365, -0.0011060, 0.0011163
6: 0.0093209, 0.0098908, 0.0092964, 0.0099311, -0.0004580, 0.0005639
7: -0.0191781, -0.0158998, -0.0193192, -0.0156676, -0.0024059, 0.0023487
8: 0.9688433, 0.9782360, 0.9684391, 0.9789013, -0.0069566, 0.0069270
9: 0.0040496, 0.0068101, 0.0038541, 0.0069289, -0.0019932, 0.0020318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052190
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052203
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002768, 0.0000301, -0.0002897, 0.0000900, -0.0002634, 0.0002220
1: 0.0000242, 0.0013677, -0.0000361, 0.0014596, -0.0009950, 0.0009985
2: 0.0142916, 0.0163038, 0.0141541, 0.0163941, -0.0014834, 0.0014864
3: 0.0001198, 0.0016329, 0.0000164, 0.0017008, -0.0011102, 0.0011161
4: -0.0042691, -0.0028735, -0.0043645, -0.0028109, -0.0010657, 0.0010425
5: 0.0080578, 0.0095681, 0.0079545, 0.0096358, -0.0011077, 0.0011139
6: 0.0093222, 0.0098921, 0.0092966, 0.0099311, -0.0004619, 0.0005643
7: -0.0191707, -0.0158920, -0.0193178, -0.0156678, -0.0023991, 0.0023520
8: 0.9688647, 0.9782584, 0.9684432, 0.9789007, -0.0069432, 0.0069377
9: 0.0040430, 0.0068038, 0.0038542, 0.0069277, -0.0019962, 0.0020266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052190
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052203
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002660, 0.0000687, -0.0002970, 0.0000348, -0.0002051, 0.0002717
1: 0.0000748, 0.0014269, -0.0000701, 0.0013750, -0.0009009, 0.0010941
2: 0.0142031, 0.0162280, 0.0142809, 0.0164450, -0.0016356, 0.0013391
3: 0.0000532, 0.0015759, 0.0001117, 0.0017390, -0.0012286, 0.0010027
4: -0.0043305, -0.0029260, -0.0042766, -0.0027756, -0.0011431, 0.0009635
5: 0.0079913, 0.0095112, 0.0080497, 0.0096740, -0.0012263, 0.0010005
6: 0.0093437, 0.0099172, 0.0092822, 0.0098952, -0.0005004, 0.0004943
7: -0.0190472, -0.0157477, -0.0194007, -0.0158744, -0.0021268, 0.0026462
8: 0.9692186, 0.9786719, 0.9682056, 0.9783089, -0.0062617, 0.0076390
9: 0.0039215, 0.0066999, 0.0040282, 0.0069975, -0.0022337, 0.0018053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0053497
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0053497
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002655, 0.0000713, -0.0002969, 0.0000348, -0.0002045, 0.0002741
1: 0.0000771, 0.0014308, -0.0000696, 0.0013749, -0.0009002, 0.0010989
2: 0.0141971, 0.0162246, 0.0142809, 0.0164442, -0.0016429, 0.0013371
3: 0.0000487, 0.0015733, 0.0001117, 0.0017385, -0.0012340, 0.0010009
4: -0.0043347, -0.0029284, -0.0042766, -0.0027761, -0.0011481, 0.0009642
5: 0.0079868, 0.0095086, 0.0080497, 0.0096735, -0.0012317, 0.0009986
6: 0.0093446, 0.0099189, 0.0092824, 0.0098952, -0.0005065, 0.0004964
7: -0.0190416, -0.0157380, -0.0193995, -0.0158744, -0.0021199, 0.0026578
8: 0.9692345, 0.9786997, 0.9682090, 0.9783087, -0.0062532, 0.0076727
9: 0.0039133, 0.0066951, 0.0040282, 0.0069965, -0.0022434, 0.0018005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0053501
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0053501
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002660, 0.0000687, -0.0002990, 0.0000424, -0.0002123, 0.0002738
1: 0.0000748, 0.0014269, -0.0000793, 0.0013866, -0.0009124, 0.0011083
2: 0.0142031, 0.0162280, 0.0142634, 0.0164588, -0.0016553, 0.0013564
3: 0.0000532, 0.0015759, 0.0000986, 0.0017494, -0.0012428, 0.0010157
4: -0.0043305, -0.0029260, -0.0042887, -0.0027660, -0.0011616, 0.0009754
5: 0.0079913, 0.0095112, 0.0080366, 0.0096844, -0.0012403, 0.0010134
6: 0.0093437, 0.0099172, 0.0092783, 0.0099001, -0.0005053, 0.0005126
7: -0.0190472, -0.0157477, -0.0194232, -0.0158460, -0.0021549, 0.0026712
8: 0.9692186, 0.9786719, 0.9681412, 0.9783903, -0.0063423, 0.0077323
9: 0.0039215, 0.0066999, 0.0040043, 0.0070165, -0.0022561, 0.0018291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0053095
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0053095
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002655, 0.0000713, -0.0002989, 0.0000423, -0.0002116, 0.0002761
1: 0.0000771, 0.0014308, -0.0000789, 0.0013865, -0.0009117, 0.0011127
2: 0.0141971, 0.0162246, 0.0142635, 0.0164582, -0.0016619, 0.0013544
3: 0.0000487, 0.0015733, 0.0000986, 0.0017489, -0.0012477, 0.0010138
4: -0.0043347, -0.0029284, -0.0042886, -0.0027664, -0.0011664, 0.0009762
5: 0.0079868, 0.0095086, 0.0080366, 0.0096839, -0.0012453, 0.0010116
6: 0.0093446, 0.0099189, 0.0092785, 0.0099001, -0.0005113, 0.0005145
7: -0.0190416, -0.0157380, -0.0194222, -0.0158461, -0.0021480, 0.0026819
8: 0.9692345, 0.9786997, 0.9681441, 0.9783898, -0.0063337, 0.0077630
9: 0.0039133, 0.0066951, 0.0040044, 0.0070156, -0.0022651, 0.0018242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0053095
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0053095
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0000753, -0.0002970, 0.0000348, -0.0002083, 0.0002781
1: 0.0000653, 0.0014370, -0.0000701, 0.0013750, -0.0009198, 0.0011049
2: 0.0141879, 0.0162421, 0.0142809, 0.0164450, -0.0016519, 0.0013676
3: 0.0000418, 0.0015865, 0.0001117, 0.0017390, -0.0012409, 0.0010239
4: -0.0043411, -0.0029163, -0.0042766, -0.0027756, -0.0011544, 0.0009827
5: 0.0079799, 0.0095218, 0.0080497, 0.0096740, -0.0012385, 0.0010216
6: 0.0093397, 0.0099215, 0.0092822, 0.0098952, -0.0005170, 0.0004989
7: -0.0190702, -0.0157230, -0.0194007, -0.0158744, -0.0021659, 0.0026727
8: 0.9691527, 0.9787427, 0.9682056, 0.9783089, -0.0063949, 0.0077149
9: 0.0039007, 0.0067192, 0.0040282, 0.0069975, -0.0022560, 0.0018411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054052
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054052
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002676, 0.0000775, -0.0002969, 0.0000348, -0.0002076, 0.0002801
1: 0.0000674, 0.0014404, -0.0000696, 0.0013749, -0.0009186, 0.0011093
2: 0.0141828, 0.0162391, 0.0142809, 0.0164442, -0.0016585, 0.0013654
3: 0.0000380, 0.0015842, 0.0001117, 0.0017385, -0.0012458, 0.0010220
4: -0.0043446, -0.0029184, -0.0042766, -0.0027761, -0.0011590, 0.0009829
5: 0.0079761, 0.0095195, 0.0080497, 0.0096735, -0.0012434, 0.0010197
6: 0.0093405, 0.0099230, 0.0092824, 0.0098952, -0.0005222, 0.0005008
7: -0.0190652, -0.0157146, -0.0193995, -0.0158744, -0.0021592, 0.0026833
8: 0.9691668, 0.9787666, 0.9682090, 0.9783087, -0.0063849, 0.0077458
9: 0.0038936, 0.0067151, 0.0040282, 0.0069965, -0.0022649, 0.0018362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054055
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054055
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0000753, -0.0002990, 0.0000424, -0.0002133, 0.0002789
1: 0.0000653, 0.0014370, -0.0000793, 0.0013866, -0.0009240, 0.0011135
2: 0.0141879, 0.0162421, 0.0142634, 0.0164588, -0.0016640, 0.0013726
3: 0.0000418, 0.0015865, 0.0000986, 0.0017494, -0.0012496, 0.0010272
4: -0.0043411, -0.0029163, -0.0042887, -0.0027660, -0.0011652, 0.0009899
5: 0.0079799, 0.0095218, 0.0080366, 0.0096844, -0.0012472, 0.0010248
6: 0.0093397, 0.0099215, 0.0092783, 0.0099001, -0.0005252, 0.0005157
7: -0.0190702, -0.0157230, -0.0194232, -0.0158460, -0.0021719, 0.0026877
8: 0.9691527, 0.9787427, 0.9681412, 0.9783903, -0.0064195, 0.0077723
9: 0.0039007, 0.0067192, 0.0040043, 0.0070165, -0.0022699, 0.0018454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054052
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054052
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002676, 0.0000775, -0.0002989, 0.0000423, -0.0002127, 0.0002810
1: 0.0000674, 0.0014404, -0.0000789, 0.0013865, -0.0009232, 0.0011179
2: 0.0141828, 0.0162391, 0.0142635, 0.0164582, -0.0016706, 0.0013708
3: 0.0000380, 0.0015842, 0.0000986, 0.0017489, -0.0012546, 0.0010255
4: -0.0043446, -0.0029184, -0.0042886, -0.0027664, -0.0011698, 0.0009906
5: 0.0079761, 0.0095195, 0.0080366, 0.0096839, -0.0012521, 0.0010232
6: 0.0093405, 0.0099230, 0.0092785, 0.0099001, -0.0005305, 0.0005174
7: -0.0190652, -0.0157146, -0.0194222, -0.0158461, -0.0021657, 0.0026984
8: 0.9691668, 0.9787666, 0.9681441, 0.9783898, -0.0064113, 0.0078032
9: 0.0038936, 0.0067151, 0.0040044, 0.0070156, -0.0022789, 0.0018410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054055
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054055
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002660, 0.0000687, -0.0002876, 0.0000835, -0.0002430, 0.0002535
1: 0.0000748, 0.0014269, -0.0000264, 0.0014496, -0.0009337, 0.0010218
2: 0.0142031, 0.0162280, 0.0141690, 0.0163795, -0.0015219, 0.0013888
3: 0.0000532, 0.0015759, 0.0000276, 0.0016898, -0.0011407, 0.0010401
4: -0.0043305, -0.0029260, -0.0043542, -0.0028210, -0.0010862, 0.0009997
5: 0.0079913, 0.0095112, 0.0079657, 0.0096249, -0.0011383, 0.0010378
6: 0.0093437, 0.0099172, 0.0093008, 0.0099269, -0.0005264, 0.0005663
7: -0.0190472, -0.0157477, -0.0192940, -0.0156922, -0.0022089, 0.0024313
8: 0.9692186, 0.9786719, 0.9685115, 0.9788310, -0.0064929, 0.0071132
9: 0.0039215, 0.0066999, 0.0038747, 0.0069077, -0.0020604, 0.0018744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051158, upper bound: 0.0054308
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051158, upper bound: 0.0054312
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002655, 0.0000713, -0.0002875, 0.0000835, -0.0002425, 0.0002556
1: 0.0000771, 0.0014308, -0.0000258, 0.0014496, -0.0009339, 0.0010251
2: 0.0141971, 0.0162246, 0.0141690, 0.0163787, -0.0015266, 0.0013883
3: 0.0000487, 0.0015733, 0.0000276, 0.0016892, -0.0011442, 0.0010396
4: -0.0043347, -0.0029284, -0.0043542, -0.0028215, -0.0010897, 0.0010024
5: 0.0079868, 0.0095086, 0.0079657, 0.0096243, -0.0011418, 0.0010372
6: 0.0093446, 0.0099189, 0.0093010, 0.0099269, -0.0005329, 0.0005673
7: -0.0190416, -0.0157380, -0.0192927, -0.0156922, -0.0022046, 0.0024386
8: 0.9692345, 0.9786997, 0.9685151, 0.9788309, -0.0064919, 0.0071353
9: 0.0039133, 0.0066951, 0.0038748, 0.0069066, -0.0020668, 0.0018717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051158, upper bound: 0.0054308
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051158, upper bound: 0.0054312
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002660, 0.0000687, -0.0002898, 0.0000901, -0.0002493, 0.0002558
1: 0.0000748, 0.0014269, -0.0000367, 0.0014597, -0.0009448, 0.0010431
2: 0.0142031, 0.0162280, 0.0141540, 0.0163950, -0.0015505, 0.0014055
3: 0.0000532, 0.0015759, 0.0000163, 0.0017014, -0.0011609, 0.0010527
4: -0.0043305, -0.0029260, -0.0043646, -0.0028102, -0.0011116, 0.0010113
5: 0.0079913, 0.0095112, 0.0079544, 0.0096365, -0.0011583, 0.0010504
6: 0.0093437, 0.0099172, 0.0092964, 0.0099311, -0.0005311, 0.0005826
7: -0.0190472, -0.0157477, -0.0193192, -0.0156676, -0.0022361, 0.0024602
8: 0.9692186, 0.9786719, 0.9684391, 0.9789013, -0.0065709, 0.0072503
9: 0.0039215, 0.0066999, 0.0038541, 0.0069289, -0.0020893, 0.0018973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0053837
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0053837
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002655, 0.0000713, -0.0002897, 0.0000900, -0.0002488, 0.0002579
1: 0.0000771, 0.0014308, -0.0000361, 0.0014596, -0.0009450, 0.0010458
2: 0.0141971, 0.0162246, 0.0141541, 0.0163941, -0.0015548, 0.0014050
3: 0.0000487, 0.0015733, 0.0000164, 0.0017008, -0.0011641, 0.0010521
4: -0.0043347, -0.0029284, -0.0043645, -0.0028109, -0.0011144, 0.0010139
5: 0.0079868, 0.0095086, 0.0079545, 0.0096358, -0.0011615, 0.0010497
6: 0.0093446, 0.0099189, 0.0092966, 0.0099311, -0.0005376, 0.0005835
7: -0.0190416, -0.0157380, -0.0193178, -0.0156678, -0.0022317, 0.0024669
8: 0.9692345, 0.9786997, 0.9684432, 0.9789007, -0.0065695, 0.0072702
9: 0.0039133, 0.0066951, 0.0038542, 0.0069277, -0.0020950, 0.0018945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0053842
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0053842
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0000753, -0.0002876, 0.0000835, -0.0002473, 0.0002611
1: 0.0000653, 0.0014370, -0.0000264, 0.0014496, -0.0009579, 0.0010397
2: 0.0141879, 0.0162421, 0.0141690, 0.0163795, -0.0015486, 0.0014237
3: 0.0000418, 0.0015865, 0.0000276, 0.0016898, -0.0011609, 0.0010657
4: -0.0043411, -0.0029163, -0.0043542, -0.0028210, -0.0011048, 0.0010234
5: 0.0079799, 0.0095218, 0.0079657, 0.0096249, -0.0011584, 0.0010633
6: 0.0093397, 0.0099215, 0.0093008, 0.0099269, -0.0005443, 0.0005739
7: -0.0190702, -0.0157230, -0.0192940, -0.0156922, -0.0022589, 0.0024749
8: 0.9691527, 0.9787427, 0.9685115, 0.9788310, -0.0066576, 0.0072383
9: 0.0039007, 0.0067192, 0.0038747, 0.0069077, -0.0020972, 0.0019168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050848, upper bound: 0.0055035
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050848, upper bound: 0.0055035
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002676, 0.0000775, -0.0002875, 0.0000835, -0.0002469, 0.0002629
1: 0.0000674, 0.0014404, -0.0000258, 0.0014496, -0.0009579, 0.0010437
2: 0.0141828, 0.0162391, 0.0141690, 0.0163787, -0.0015544, 0.0014232
3: 0.0000380, 0.0015842, 0.0000276, 0.0016892, -0.0011652, 0.0010651
4: -0.0043446, -0.0029184, -0.0043542, -0.0028215, -0.0011090, 0.0010251
5: 0.0079761, 0.0095195, 0.0079657, 0.0096243, -0.0011627, 0.0010626
6: 0.0093405, 0.0099230, 0.0093010, 0.0099269, -0.0005495, 0.0005752
7: -0.0190652, -0.0157146, -0.0192927, -0.0156922, -0.0022547, 0.0024840
8: 0.9691668, 0.9787666, 0.9685151, 0.9788309, -0.0066559, 0.0072652
9: 0.0038936, 0.0067151, 0.0038748, 0.0069066, -0.0021050, 0.0019140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050848, upper bound: 0.0055046
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050848, upper bound: 0.0055046
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0000753, -0.0002898, 0.0000901, -0.0002504, 0.0002610
1: 0.0000653, 0.0014370, -0.0000367, 0.0014597, -0.0009584, 0.0010431
2: 0.0141879, 0.0162421, 0.0141540, 0.0163950, -0.0015513, 0.0014221
3: 0.0000418, 0.0015865, 0.0000163, 0.0017014, -0.0011625, 0.0010638
4: -0.0043411, -0.0029163, -0.0043646, -0.0028102, -0.0011126, 0.0010295
5: 0.0079799, 0.0095218, 0.0079544, 0.0096365, -0.0011600, 0.0010614
6: 0.0093397, 0.0099215, 0.0092964, 0.0099311, -0.0005530, 0.0005916
7: -0.0190702, -0.0157230, -0.0193192, -0.0156676, -0.0022509, 0.0024734
8: 0.9691527, 0.9787427, 0.9684391, 0.9789013, -0.0066516, 0.0072530
9: 0.0039007, 0.0067192, 0.0038541, 0.0069289, -0.0020974, 0.0019120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050877, upper bound: 0.0055035
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050877, upper bound: 0.0055035
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002676, 0.0000775, -0.0002897, 0.0000900, -0.0002498, 0.0002629
1: 0.0000674, 0.0014404, -0.0000361, 0.0014596, -0.0009582, 0.0010460
2: 0.0141828, 0.0162391, 0.0141541, 0.0163941, -0.0015556, 0.0014209
3: 0.0000380, 0.0015842, 0.0000164, 0.0017008, -0.0011656, 0.0010626
4: -0.0043446, -0.0029184, -0.0043645, -0.0028109, -0.0011156, 0.0010312
5: 0.0079761, 0.0095195, 0.0079545, 0.0096358, -0.0011631, 0.0010601
6: 0.0093405, 0.0099230, 0.0092966, 0.0099311, -0.0005582, 0.0005924
7: -0.0190652, -0.0157146, -0.0193178, -0.0156678, -0.0022451, 0.0024800
8: 0.9691668, 0.9787666, 0.9684432, 0.9789007, -0.0066471, 0.0072735
9: 0.0038936, 0.0067151, 0.0038542, 0.0069277, -0.0021030, 0.0019079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050877, upper bound: 0.0055046
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050877, upper bound: 0.0055046
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002938, 0.0000301, -0.0002787, 0.0000264, -0.0002140, 0.0002078
1: -0.0000554, 0.0013677, 0.0000155, 0.0013620, -0.0009391, 0.0009042
2: 0.0142916, 0.0164230, 0.0143002, 0.0163167, -0.0013511, 0.0014041
3: 0.0001198, 0.0017225, 0.0001262, 0.0016426, -0.0010146, 0.0010547
4: -0.0042691, -0.0027908, -0.0042632, -0.0028645, -0.0009463, 0.0009811
5: 0.0080578, 0.0096575, 0.0080642, 0.0095778, -0.0010126, 0.0010527
6: 0.0092884, 0.0098921, 0.0093185, 0.0098897, -0.0004222, 0.0004145
7: -0.0193648, -0.0158920, -0.0191917, -0.0159059, -0.0022720, 0.0021823
8: 0.9683084, 0.9782585, 0.9688044, 0.9782186, -0.0065575, 0.0063108
9: 0.0040430, 0.0069673, 0.0040547, 0.0068216, -0.0018429, 0.0019176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051919, upper bound: 0.0048901
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051919, upper bound: 0.0048901
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002937, 0.0000323, -0.0002786, 0.0000264, -0.0002137, 0.0002127
1: -0.0000548, 0.0013712, 0.0000161, 0.0013621, -0.0009376, 0.0009219
2: 0.0142865, 0.0164220, 0.0143001, 0.0163159, -0.0013777, 0.0014018
3: 0.0001159, 0.0017218, 0.0001261, 0.0016420, -0.0010346, 0.0010529
4: -0.0042727, -0.0027915, -0.0042633, -0.0028651, -0.0009647, 0.0009798
5: 0.0080539, 0.0096568, 0.0080641, 0.0095771, -0.0010326, 0.0010509
6: 0.0092887, 0.0098936, 0.0093188, 0.0098897, -0.0004243, 0.0004221
7: -0.0193633, -0.0158836, -0.0191903, -0.0159057, -0.0022676, 0.0022257
8: 0.9683129, 0.9782826, 0.9688083, 0.9782192, -0.0065467, 0.0064347
9: 0.0040359, 0.0069660, 0.0040545, 0.0068204, -0.0018795, 0.0019141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0048946
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0048946
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002938, 0.0000301, -0.0002691, 0.0000735, -0.0002683, 0.0002067
1: -0.0000554, 0.0013677, 0.0000604, 0.0014342, -0.0010600, 0.0009256
2: 0.0142916, 0.0164230, 0.0141921, 0.0162495, -0.0013740, 0.0015850
3: 0.0001198, 0.0017225, 0.0000450, 0.0015920, -0.0010281, 0.0011908
4: -0.0042691, -0.0027908, -0.0043381, -0.0029112, -0.0009917, 0.0011066
5: 0.0080578, 0.0096575, 0.0079831, 0.0095273, -0.0010258, 0.0011885
6: 0.0092884, 0.0098921, 0.0093376, 0.0099203, -0.0004735, 0.0005220
7: -0.0193648, -0.0158920, -0.0190821, -0.0157298, -0.0025669, 0.0021783
8: 0.9683084, 0.9782585, 0.9691182, 0.9787231, -0.0074023, 0.0064272
9: 0.0040430, 0.0069673, 0.0039064, 0.0067293, -0.0018491, 0.0021659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053497, upper bound: 0.0048901
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053497, upper bound: 0.0048901
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002937, 0.0000323, -0.0002690, 0.0000735, -0.0002679, 0.0002117
1: -0.0000548, 0.0013712, 0.0000610, 0.0014343, -0.0010585, 0.0009435
2: 0.0142865, 0.0164220, 0.0141920, 0.0162486, -0.0014008, 0.0015827
3: 0.0001159, 0.0017218, 0.0000449, 0.0015914, -0.0010482, 0.0011890
4: -0.0042727, -0.0027915, -0.0043382, -0.0029118, -0.0010104, 0.0011053
5: 0.0080539, 0.0096568, 0.0079830, 0.0095266, -0.0010458, 0.0011867
6: 0.0092887, 0.0098936, 0.0093378, 0.0099204, -0.0004756, 0.0005303
7: -0.0193633, -0.0158836, -0.0190807, -0.0157296, -0.0025625, 0.0022212
8: 0.9683129, 0.9782826, 0.9691223, 0.9787237, -0.0073916, 0.0065526
9: 0.0040359, 0.0069660, 0.0039063, 0.0067281, -0.0018856, 0.0021624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053497, upper bound: 0.0048946
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053497, upper bound: 0.0048946
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0000377, -0.0002787, 0.0000264, -0.0002169, 0.0002148
1: -0.0000642, 0.0013794, 0.0000155, 0.0013620, -0.0009561, 0.0009154
2: 0.0142741, 0.0164362, 0.0143002, 0.0163167, -0.0013679, 0.0014282
3: 0.0001066, 0.0017324, 0.0001262, 0.0016426, -0.0010272, 0.0010722
4: -0.0042813, -0.0027817, -0.0042632, -0.0028645, -0.0009580, 0.0010019
5: 0.0080446, 0.0096674, 0.0080642, 0.0095778, -0.0010252, 0.0010701
6: 0.0092847, 0.0098971, 0.0093185, 0.0098897, -0.0004412, 0.0004192
7: -0.0193864, -0.0158634, -0.0191917, -0.0159059, -0.0023061, 0.0022097
8: 0.9682467, 0.9783403, 0.9688044, 0.9782186, -0.0066713, 0.0063893
9: 0.0040189, 0.0069855, 0.0040547, 0.0068216, -0.0018660, 0.0019474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049227
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049227
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0000400, -0.0002786, 0.0000264, -0.0002165, 0.0002192
1: -0.0000640, 0.0013830, 0.0000161, 0.0013621, -0.0009546, 0.0009313
2: 0.0142688, 0.0164358, 0.0143001, 0.0163159, -0.0013918, 0.0014259
3: 0.0001027, 0.0017322, 0.0001261, 0.0016420, -0.0010452, 0.0010704
4: -0.0042849, -0.0027819, -0.0042633, -0.0028651, -0.0009745, 0.0010010
5: 0.0080406, 0.0096672, 0.0080641, 0.0095771, -0.0010432, 0.0010683
6: 0.0092848, 0.0098986, 0.0093188, 0.0098897, -0.0004433, 0.0004261
7: -0.0193858, -0.0158548, -0.0191903, -0.0159057, -0.0023012, 0.0022487
8: 0.9682482, 0.9783649, 0.9688083, 0.9782192, -0.0066607, 0.0065006
9: 0.0040117, 0.0069850, 0.0040545, 0.0068204, -0.0018988, 0.0019436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0000377, -0.0002691, 0.0000735, -0.0002711, 0.0002137
1: -0.0000642, 0.0013794, 0.0000604, 0.0014342, -0.0010769, 0.0009368
2: 0.0142741, 0.0164362, 0.0141921, 0.0162495, -0.0013909, 0.0016091
3: 0.0001066, 0.0017324, 0.0000450, 0.0015920, -0.0010408, 0.0012083
4: -0.0042813, -0.0027817, -0.0043381, -0.0029112, -0.0010034, 0.0011274
5: 0.0080446, 0.0096674, 0.0079831, 0.0095273, -0.0010385, 0.0012059
6: 0.0092847, 0.0098971, 0.0093376, 0.0099203, -0.0004924, 0.0005268
7: -0.0193864, -0.0158634, -0.0190821, -0.0157298, -0.0026009, 0.0022057
8: 0.9682467, 0.9783403, 0.9691182, 0.9787231, -0.0075161, 0.0065057
9: 0.0040189, 0.0069855, 0.0039064, 0.0067293, -0.0018722, 0.0021956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049227
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049227
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0000400, -0.0002690, 0.0000735, -0.0002707, 0.0002182
1: -0.0000640, 0.0013830, 0.0000610, 0.0014343, -0.0010755, 0.0009530
2: 0.0142688, 0.0164358, 0.0141920, 0.0162486, -0.0014150, 0.0016068
3: 0.0001027, 0.0017322, 0.0000449, 0.0015914, -0.0010588, 0.0012065
4: -0.0042849, -0.0027819, -0.0043382, -0.0029118, -0.0010202, 0.0011265
5: 0.0080406, 0.0096672, 0.0079830, 0.0095266, -0.0010564, 0.0012041
6: 0.0092848, 0.0098986, 0.0093378, 0.0099204, -0.0004945, 0.0005343
7: -0.0193858, -0.0158548, -0.0190807, -0.0157296, -0.0025961, 0.0022443
8: 0.9682482, 0.9783649, 0.9691223, 0.9787237, -0.0075055, 0.0066185
9: 0.0040117, 0.0069850, 0.0039063, 0.0067281, -0.0019050, 0.0021919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049228
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049228
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002938, 0.0000301, -0.0002805, 0.0000330, -0.0002232, 0.0002122
1: -0.0000554, 0.0013677, 0.0000072, 0.0013722, -0.0009653, 0.0009253
2: 0.0142916, 0.0164230, 0.0142849, 0.0163292, -0.0013825, 0.0014432
3: 0.0001198, 0.0017225, 0.0001148, 0.0016519, -0.0010382, 0.0010842
4: -0.0042691, -0.0027908, -0.0042738, -0.0028559, -0.0009704, 0.0010082
5: 0.0080578, 0.0096575, 0.0080527, 0.0095871, -0.0010362, 0.0010821
6: 0.0092884, 0.0098921, 0.0093150, 0.0098940, -0.0004333, 0.0004333
7: -0.0193648, -0.0158920, -0.0192120, -0.0158810, -0.0023358, 0.0022334
8: 0.9683084, 0.9782585, 0.9687462, 0.9782898, -0.0067403, 0.0064574
9: 0.0040430, 0.0069673, 0.0040338, 0.0068386, -0.0018859, 0.0019714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052190, upper bound: 0.0048592
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052190, upper bound: 0.0048592
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002937, 0.0000323, -0.0002803, 0.0000331, -0.0002228, 0.0002171
1: -0.0000548, 0.0013712, 0.0000079, 0.0013723, -0.0009638, 0.0009429
2: 0.0142865, 0.0164220, 0.0142848, 0.0163282, -0.0014088, 0.0014409
3: 0.0001159, 0.0017218, 0.0001147, 0.0016512, -0.0010580, 0.0010824
4: -0.0042727, -0.0027915, -0.0042739, -0.0028565, -0.0009886, 0.0010069
5: 0.0080539, 0.0096568, 0.0080526, 0.0095864, -0.0010560, 0.0010803
6: 0.0092887, 0.0098936, 0.0093153, 0.0098941, -0.0004354, 0.0004411
7: -0.0193633, -0.0158836, -0.0192104, -0.0158809, -0.0023314, 0.0022763
8: 0.9683129, 0.9782826, 0.9687508, 0.9782904, -0.0067294, 0.0065805
9: 0.0040359, 0.0069660, 0.0040336, 0.0068373, -0.0019221, 0.0019678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052200, upper bound: 0.0048668
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052200, upper bound: 0.0048668
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002938, 0.0000301, -0.0002711, 0.0000801, -0.0002748, 0.0002094
1: -0.0000554, 0.0013677, 0.0000509, 0.0014444, -0.0010721, 0.0009425
2: 0.0142916, 0.0164230, 0.0141769, 0.0162638, -0.0014000, 0.0016033
3: 0.0001198, 0.0017225, 0.0000335, 0.0016028, -0.0010477, 0.0012045
4: -0.0042691, -0.0027908, -0.0043487, -0.0029012, -0.0010098, 0.0011192
5: 0.0080578, 0.0096575, 0.0079716, 0.0095380, -0.0010453, 0.0012022
6: 0.0092884, 0.0098921, 0.0093335, 0.0099246, -0.0004786, 0.0005386
7: -0.0193648, -0.0158920, -0.0191054, -0.0157050, -0.0025965, 0.0022117
8: 0.9683084, 0.9782585, 0.9690516, 0.9787943, -0.0074873, 0.0065477
9: 0.0040430, 0.0069673, 0.0038855, 0.0067489, -0.0018810, 0.0021909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054052, upper bound: 0.0048592
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054052, upper bound: 0.0048592
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002937, 0.0000323, -0.0002710, 0.0000801, -0.0002744, 0.0002143
1: -0.0000548, 0.0013712, 0.0000514, 0.0014444, -0.0010706, 0.0009604
2: 0.0142865, 0.0164220, 0.0141768, 0.0162629, -0.0014268, 0.0016009
3: 0.0001159, 0.0017218, 0.0000334, 0.0016021, -0.0010677, 0.0012027
4: -0.0042727, -0.0027915, -0.0043488, -0.0029018, -0.0010286, 0.0011179
5: 0.0080539, 0.0096568, 0.0079715, 0.0095374, -0.0010653, 0.0012004
6: 0.0092887, 0.0098936, 0.0093338, 0.0099247, -0.0004807, 0.0005467
7: -0.0193633, -0.0158836, -0.0191041, -0.0157048, -0.0025921, 0.0022549
8: 0.9683129, 0.9782826, 0.9690555, 0.9787948, -0.0074764, 0.0066730
9: 0.0040359, 0.0069660, 0.0038854, 0.0067478, -0.0019175, 0.0021873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054052, upper bound: 0.0048668
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054052, upper bound: 0.0048668
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0000377, -0.0002805, 0.0000330, -0.0002218, 0.0002156
1: -0.0000642, 0.0013794, 0.0000072, 0.0013722, -0.0009592, 0.0009246
2: 0.0142741, 0.0164362, 0.0142849, 0.0163292, -0.0013807, 0.0014336
3: 0.0001066, 0.0017324, 0.0001148, 0.0016519, -0.0010364, 0.0010766
4: -0.0042813, -0.0027817, -0.0042738, -0.0028559, -0.0009702, 0.0010036
5: 0.0080446, 0.0096674, 0.0080527, 0.0095871, -0.0010343, 0.0010745
6: 0.0092847, 0.0098971, 0.0093150, 0.0098940, -0.0004434, 0.0004365
7: -0.0193864, -0.0158634, -0.0192120, -0.0158810, -0.0023163, 0.0022239
8: 0.9682467, 0.9783403, 0.9687462, 0.9782898, -0.0066957, 0.0064499
9: 0.0040189, 0.0069855, 0.0040338, 0.0068386, -0.0018797, 0.0019559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049227
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049227
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0000400, -0.0002803, 0.0000331, -0.0002215, 0.0002205
1: -0.0000640, 0.0013830, 0.0000079, 0.0013723, -0.0009578, 0.0009419
2: 0.0142688, 0.0164358, 0.0142848, 0.0163282, -0.0014067, 0.0014313
3: 0.0001027, 0.0017322, 0.0001147, 0.0016512, -0.0010560, 0.0010749
4: -0.0042849, -0.0027819, -0.0042739, -0.0028565, -0.0009884, 0.0010024
5: 0.0080406, 0.0096672, 0.0080526, 0.0095864, -0.0010539, 0.0010728
6: 0.0092848, 0.0098986, 0.0093153, 0.0098941, -0.0004461, 0.0004440
7: -0.0193858, -0.0158548, -0.0192104, -0.0158809, -0.0023122, 0.0022665
8: 0.9682482, 0.9783649, 0.9687508, 0.9782904, -0.0066855, 0.0065714
9: 0.0040117, 0.0069850, 0.0040336, 0.0068373, -0.0019156, 0.0019527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0000377, -0.0002711, 0.0000801, -0.0002757, 0.0002147
1: -0.0000642, 0.0013794, 0.0000509, 0.0014444, -0.0010801, 0.0009485
2: 0.0142741, 0.0164362, 0.0141769, 0.0162638, -0.0014073, 0.0016146
3: 0.0001066, 0.0017324, 0.0000335, 0.0016028, -0.0010526, 0.0012127
4: -0.0042813, -0.0027817, -0.0043487, -0.0029012, -0.0010185, 0.0011292
5: 0.0080446, 0.0096674, 0.0079716, 0.0095380, -0.0010502, 0.0012104
6: 0.0092847, 0.0098971, 0.0093335, 0.0099246, -0.0004947, 0.0005474
7: -0.0193864, -0.0158634, -0.0191054, -0.0157050, -0.0026112, 0.0022222
8: 0.9682467, 0.9783403, 0.9690516, 0.9787943, -0.0075409, 0.0065829
9: 0.0040189, 0.0069855, 0.0038855, 0.0067489, -0.0018884, 0.0022043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049227
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049227
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0000400, -0.0002710, 0.0000801, -0.0002754, 0.0002197
1: -0.0000640, 0.0013830, 0.0000514, 0.0014444, -0.0010787, 0.0009661
2: 0.0142688, 0.0164358, 0.0141768, 0.0162629, -0.0014336, 0.0016124
3: 0.0001027, 0.0017322, 0.0000334, 0.0016021, -0.0010725, 0.0012110
4: -0.0042849, -0.0027819, -0.0043488, -0.0029018, -0.0010368, 0.0011280
5: 0.0080406, 0.0096672, 0.0079715, 0.0095374, -0.0010700, 0.0012087
6: 0.0092848, 0.0098986, 0.0093338, 0.0099247, -0.0004974, 0.0005555
7: -0.0193858, -0.0158548, -0.0191041, -0.0157048, -0.0026072, 0.0022646
8: 0.9682482, 0.9783649, 0.9690555, 0.9787948, -0.0075306, 0.0067057
9: 0.0040117, 0.0069850, 0.0038854, 0.0067478, -0.0019244, 0.0022010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049228
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049228
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0000789, -0.0002787, 0.0000264, -0.0002093, 0.0002586
1: -0.0000118, 0.0014426, 0.0000155, 0.0013620, -0.0009380, 0.0010027
2: 0.0141796, 0.0163577, 0.0143002, 0.0163167, -0.0014986, 0.0013957
3: 0.0000355, 0.0016734, 0.0001262, 0.0016426, -0.0011255, 0.0010454
4: -0.0043469, -0.0028361, -0.0042632, -0.0028645, -0.0010486, 0.0009984
5: 0.0079736, 0.0096085, 0.0080642, 0.0095778, -0.0011233, 0.0010431
6: 0.0093069, 0.0099239, 0.0093185, 0.0098897, -0.0005164, 0.0004562
7: -0.0192584, -0.0157094, -0.0191917, -0.0159059, -0.0022251, 0.0024226
8: 0.9686132, 0.9787818, 0.9688044, 0.9782186, -0.0065250, 0.0069992
9: 0.0038892, 0.0068777, 0.0040547, 0.0068216, -0.0020453, 0.0018858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051919, upper bound: 0.0050783
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051919, upper bound: 0.0050783
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002842, 0.0000808, -0.0002786, 0.0000264, -0.0002089, 0.0002614
1: -0.0000103, 0.0014455, 0.0000161, 0.0013621, -0.0009363, 0.0010168
2: 0.0141751, 0.0163554, 0.0143001, 0.0163159, -0.0015198, 0.0013929
3: 0.0000322, 0.0016717, 0.0001261, 0.0016420, -0.0011415, 0.0010432
4: -0.0043499, -0.0028377, -0.0042633, -0.0028651, -0.0010633, 0.0009969
5: 0.0079703, 0.0096068, 0.0080641, 0.0095771, -0.0011393, 0.0010409
6: 0.0093076, 0.0099251, 0.0093188, 0.0098897, -0.0005191, 0.0004624
7: -0.0192548, -0.0157021, -0.0191903, -0.0159057, -0.0022194, 0.0024573
8: 0.9686236, 0.9788024, 0.9688083, 0.9782192, -0.0065123, 0.0070983
9: 0.0038831, 0.0068747, 0.0040545, 0.0068204, -0.0020745, 0.0018813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0050793
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0050792
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0000789, -0.0002805, 0.0000330, -0.0002185, 0.0002631
1: -0.0000118, 0.0014426, 0.0000072, 0.0013722, -0.0009642, 0.0010238
2: 0.0141796, 0.0163577, 0.0142849, 0.0163292, -0.0015300, 0.0014348
3: 0.0000355, 0.0016734, 0.0001148, 0.0016519, -0.0011491, 0.0010748
4: -0.0043469, -0.0028361, -0.0042738, -0.0028559, -0.0010727, 0.0010256
5: 0.0079736, 0.0096085, 0.0080527, 0.0095871, -0.0011469, 0.0010725
6: 0.0093069, 0.0099239, 0.0093150, 0.0098940, -0.0005275, 0.0004750
7: -0.0192584, -0.0157094, -0.0192120, -0.0158810, -0.0022889, 0.0024737
8: 0.9686132, 0.9787818, 0.9687462, 0.9782898, -0.0067078, 0.0071459
9: 0.0038892, 0.0068777, 0.0040338, 0.0068386, -0.0020883, 0.0019395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052190, upper bound: 0.0050605
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052190, upper bound: 0.0050605
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002842, 0.0000808, -0.0002803, 0.0000331, -0.0002180, 0.0002658
1: -0.0000103, 0.0014455, 0.0000079, 0.0013723, -0.0009624, 0.0010378
2: 0.0141751, 0.0163554, 0.0142848, 0.0163282, -0.0015510, 0.0014320
3: 0.0000322, 0.0016717, 0.0001147, 0.0016512, -0.0011649, 0.0010726
4: -0.0043499, -0.0028377, -0.0042739, -0.0028565, -0.0010872, 0.0010241
5: 0.0079703, 0.0096068, 0.0080526, 0.0095864, -0.0011627, 0.0010703
6: 0.0093076, 0.0099251, 0.0093153, 0.0098941, -0.0005302, 0.0004813
7: -0.0192548, -0.0157021, -0.0192104, -0.0158809, -0.0022832, 0.0025079
8: 0.9686236, 0.9788024, 0.9687508, 0.9782904, -0.0066950, 0.0072441
9: 0.0038831, 0.0068747, 0.0040336, 0.0068373, -0.0021172, 0.0019350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052203, upper bound: 0.0050605
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052203, upper bound: 0.0050605
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002868, 0.0000856, -0.0002787, 0.0000264, -0.0002121, 0.0002646
1: -0.0000223, 0.0014529, 0.0000155, 0.0013620, -0.0009600, 0.0010138
2: 0.0141642, 0.0163733, 0.0143002, 0.0163167, -0.0015153, 0.0014274
3: 0.0000240, 0.0016852, 0.0001262, 0.0016426, -0.0011380, 0.0010686
4: -0.0043575, -0.0028252, -0.0042632, -0.0028645, -0.0010602, 0.0010224
5: 0.0079621, 0.0096202, 0.0080642, 0.0095778, -0.0011358, 0.0010662
6: 0.0093025, 0.0099282, 0.0093185, 0.0098897, -0.0005335, 0.0004610
7: -0.0192840, -0.0156843, -0.0191917, -0.0159059, -0.0022605, 0.0024498
8: 0.9685401, 0.9788535, 0.9688044, 0.9782186, -0.0066745, 0.0070773
9: 0.0038681, 0.0068992, 0.0040547, 0.0068216, -0.0020682, 0.0019209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051272
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051272
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002864, 0.0000870, -0.0002786, 0.0000264, -0.0002117, 0.0002669
1: -0.0000205, 0.0014549, 0.0000161, 0.0013621, -0.0009580, 0.0010245
2: 0.0141611, 0.0163707, 0.0143001, 0.0163159, -0.0015313, 0.0014242
3: 0.0000217, 0.0016832, 0.0001261, 0.0016420, -0.0011501, 0.0010662
4: -0.0043596, -0.0028271, -0.0042633, -0.0028651, -0.0010712, 0.0010206
5: 0.0079598, 0.0096183, 0.0080641, 0.0095771, -0.0011479, 0.0010638
6: 0.0093033, 0.0099291, 0.0093188, 0.0098897, -0.0005346, 0.0004656
7: -0.0192796, -0.0156793, -0.0191903, -0.0159057, -0.0022547, 0.0024761
8: 0.9685525, 0.9788677, 0.9688083, 0.9782192, -0.0066600, 0.0071520
9: 0.0038639, 0.0068956, 0.0040545, 0.0068204, -0.0020903, 0.0019162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051322
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051322
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002868, 0.0000856, -0.0002805, 0.0000330, -0.0002170, 0.0002660
1: -0.0000223, 0.0014529, 0.0000072, 0.0013722, -0.0009596, 0.0010232
2: 0.0141642, 0.0163733, 0.0142849, 0.0163292, -0.0015284, 0.0014270
3: 0.0000240, 0.0016852, 0.0001148, 0.0016519, -0.0011474, 0.0010685
4: -0.0043575, -0.0028252, -0.0042738, -0.0028559, -0.0010726, 0.0010232
5: 0.0079621, 0.0096202, 0.0080527, 0.0095871, -0.0011452, 0.0010661
6: 0.0093025, 0.0099282, 0.0093150, 0.0098940, -0.0005407, 0.0004783
7: -0.0192840, -0.0156843, -0.0192120, -0.0158810, -0.0022683, 0.0024645
8: 0.9685401, 0.9788535, 0.9687462, 0.9782898, -0.0066721, 0.0071392
9: 0.0038681, 0.0068992, 0.0040338, 0.0068386, -0.0020823, 0.0019245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051272
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051272
time: 0.69 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.09 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0048901
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0048901
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0048946
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0048946
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050833, upper bound: 0.0048901
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050833, upper bound: 0.0048901
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050833, upper bound: 0.0048946
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050833, upper bound: 0.0048946
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049227
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049227
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049228
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049228
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049227
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049227
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049228
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049228
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0048592
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0048592
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0048668
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0048668
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051343, upper bound: 0.0048592
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051343, upper bound: 0.0048592
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051343, upper bound: 0.0048668
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051343, upper bound: 0.0048668
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049227
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049227
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049228
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0049228
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049227
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049227
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049228
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050613, upper bound: 0.0049228
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0050833
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0050833
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0050833
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0050833
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0050613
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0050613
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0050613
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0050613
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051343
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051343
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051355
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051355
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051343
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051343
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051355
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0051355
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051271, upper bound: 0.0051602
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051271, upper bound: 0.0051602
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051271, upper bound: 0.0051610
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051271, upper bound: 0.0051610
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051923, upper bound: 0.0051497
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051923, upper bound: 0.0051497
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051923, upper bound: 0.0051504
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051923, upper bound: 0.0051504
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050968, upper bound: 0.0052356
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050968, upper bound: 0.0052356
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050968, upper bound: 0.0052382
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050968, upper bound: 0.0052382
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051011, upper bound: 0.0052356
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051011, upper bound: 0.0052356
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051011, upper bound: 0.0052382
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051011, upper bound: 0.0052382
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0051919
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0051920
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0051919
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0051920
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050783, upper bound: 0.0051919
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050783, upper bound: 0.0051920
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050783, upper bound: 0.0051919
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050783, upper bound: 0.0051920
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052190
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052200
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052190
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052200
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052190
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052203
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052190
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052203
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051272, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051272, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051272, upper bound: 0.0051443
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051272, upper bound: 0.0051443
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052190
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052200
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052190
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0052200
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052190
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052203
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052190
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050605, upper bound: 0.0052203
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0053497
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0053497
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0053501
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048901, upper bound: 0.0053501
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0053095
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0053095
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0053095
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0049227, upper bound: 0.0053095
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054052
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054052
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054055
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054055
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054052
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054052
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054055
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0048592, upper bound: 0.0054055
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051158, upper bound: 0.0054308
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051158, upper bound: 0.0054312
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051158, upper bound: 0.0054308
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051158, upper bound: 0.0054312
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0053837
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0053837
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0053842
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0053842
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050848, upper bound: 0.0055035
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050848, upper bound: 0.0055035
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050848, upper bound: 0.0055046
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050848, upper bound: 0.0055046
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050877, upper bound: 0.0055035
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050877, upper bound: 0.0055035
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050877, upper bound: 0.0055046
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0050877, upper bound: 0.0055046
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051919, upper bound: 0.0048901
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051919, upper bound: 0.0048901
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0048946
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0048946
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053497, upper bound: 0.0048901
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053497, upper bound: 0.0048901
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053497, upper bound: 0.0048946
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053497, upper bound: 0.0048946
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049227
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049227
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049227
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049227
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049228
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049228
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0052190, upper bound: 0.0048592
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0052190, upper bound: 0.0048592
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0052200, upper bound: 0.0048668
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0052200, upper bound: 0.0048668
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0054052, upper bound: 0.0048592
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0054052, upper bound: 0.0048592
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0054052, upper bound: 0.0048668
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0054052, upper bound: 0.0048668
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049227
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049227
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0049228
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049227
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049227
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049228
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0053095, upper bound: 0.0049228
IS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051919, upper bound: 0.0050783
IS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051919, upper bound: 0.0050783
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0050793
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051920, upper bound: 0.0050792
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0052190, upper bound: 0.0050605
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0052190, upper bound: 0.0050605
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0052203, upper bound: 0.0050605
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0052203, upper bound: 0.0050605
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051272
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051272
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051322
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051322
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051272
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051272
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051443, upper bound: 0.0051322
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051471
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051471
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052281
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052281
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051986, upper bound: 0.0049294
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051966, upper bound: 0.0049338
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051986, upper bound: 0.0049294
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051966, upper bound: 0.0049338
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0052246, upper bound: 0.0049007
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0052249, upper bound: 0.0049109
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0052246, upper bound: 0.0049007
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0052249, upper bound: 0.0049109
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0049618
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0049645
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051929, upper bound: 0.0050652
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051917, upper bound: 0.0050652
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051929, upper bound: 0.0050652
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051917, upper bound: 0.0050652
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0051309
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0051360
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051507, upper bound: 0.0051309
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0051445, upper bound: 0.0051360
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051472
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053730, upper bound: 0.0051472
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053729, upper bound: 0.0051476
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052282
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053440, upper bound: 0.0052203
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 8, lower bound: -0.0053359, upper bound: 0.0052282

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.76 + 598.49 = 601.25 seconds
