## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00066248


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0005199, 0.0108300, -0.0005199, 0.0108300, -0.0107732, 0.0107732)
1: (-0.0035763, 0.0026475, -0.0035763, 0.0026475, -0.0061091, 0.0061091)
2: (0.0061862, 0.0169451, 0.0061862, 0.0169451, -0.0107588, 0.0107588)
3: (1.0058427, 1.0071503, 1.0058427, 1.0071503, -0.0013076, 0.0013076)
4: (-0.0043956, -0.0009639, -0.0043956, -0.0009639, -0.0034317, 0.0034317)
5: (0.0035834, 0.0172965, 0.0035834, 0.0172965, -0.0132591, 0.0132590)
6: (-0.0130949, -0.0025337, -0.0130949, -0.0025337, -0.0105612, 0.0105612)
7: (-0.0176153, -0.0104164, -0.0176153, -0.0104164, -0.0071355, 0.0071355)
8: (-0.0152235, -0.0070763, -0.0152235, -0.0070763, -0.0081472, 0.0081472)
9: (-0.0058690, 0.0033552, -0.0058690, 0.0033552, -0.0092242, 0.0092242)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 1.96 = 3.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0008614, upper bound: 0.0008614

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007839, upper bound: 0.0008152
time: 1.05 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.03 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.03
Output dim: 3, lower bound: -0.0007839, upper bound: 0.0008152
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.03
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0005199, 0.0108300, -0.0005178, 0.0103949, -0.0103371, 0.0107713
1: -0.0035763, 0.0026475, -0.0035757, 0.0023879, -0.0058452, 0.0061054
2: 0.0061862, 0.0169451, 0.0066394, 0.0169192, -0.0107329, 0.0103056
3: 1.0058427, 1.0071503, 1.0058650, 1.0071342, -0.0012915, 0.0012853
4: -0.0043956, -0.0009639, -0.0043922, -0.0011136, -0.0032820, 0.0034283
5: 0.0035834, 0.0172965, 0.0035849, 0.0167398, -0.0127010, 0.0132576
6: -0.0130949, -0.0025337, -0.0126279, -0.0025368, -0.0105581, 0.0100941
7: -0.0176153, -0.0104164, -0.0174003, -0.0104205, -0.0071314, 0.0069205
8: -0.0152235, -0.0070763, -0.0151928, -0.0074051, -0.0078184, 0.0081165
9: -0.0058690, 0.0033552, -0.0054701, 0.0033438, -0.0092127, 0.0088253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804
time: 1.06 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804
time: 0.96 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0005168, 0.0099537, -0.0006745, 0.0083002, -0.0082967, 0.0100945
1: -0.0035755, 0.0021216, -0.0035763, 0.0011276, -0.0045873, 0.0055789
2: 0.0070919, 0.0168877, 0.0087860, 0.0167752, -0.0096834, 0.0081018
3: 1.0058861, 1.0071321, 1.0059195, 1.0071082, -0.0012221, 0.0012126
4: -0.0043881, -0.0012633, -0.0043727, -0.0018248, -0.0025633, 0.0031094
5: 0.0035856, 0.0161763, 0.0034678, 0.0140638, -0.0100678, 0.0122873
6: -0.0121557, -0.0025374, -0.0103848, -0.0025424, -0.0096132, 0.0078474
7: -0.0171959, -0.0104225, -0.0164270, -0.0099366, -0.0072035, 0.0059483
8: -0.0151553, -0.0077145, -0.0150110, -0.0088576, -0.0062977, 0.0072965
9: -0.0050767, 0.0033292, -0.0036056, 0.0032739, -0.0083506, 0.0069348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804
time: 0.75 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.95 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 3, lower bound: -0.0007804, upper bound: 0.0007804

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005178, 0.0103949, -0.0005178, 0.0103949, -0.0103352, 0.0103352
1: -0.0035757, 0.0023879, -0.0035757, 0.0023879, -0.0058415, 0.0058415
2: 0.0066394, 0.0169192, 0.0066394, 0.0169192, -0.0102797, 0.0102797
3: 1.0058650, 1.0071342, 1.0058650, 1.0071342, -0.0012692, 0.0012692
4: -0.0043922, -0.0011136, -0.0043922, -0.0011136, -0.0032786, 0.0032786
5: 0.0035849, 0.0167398, 0.0035849, 0.0167398, -0.0126996, 0.0126996
6: -0.0126279, -0.0025368, -0.0126279, -0.0025368, -0.0100911, 0.0100911
7: -0.0174003, -0.0104205, -0.0174003, -0.0104205, -0.0069164, 0.0069164
8: -0.0151928, -0.0074051, -0.0151928, -0.0074051, -0.0077877, 0.0077877
9: -0.0054701, 0.0033438, -0.0054701, 0.0033438, -0.0088138, 0.0088138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 3.98 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007162, upper bound: 0.0007652
time: 0.71 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007169, upper bound: 0.0007643
time: 0.90 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006745, 0.0083002, -0.0005178, 0.0103949, -0.0105303, 0.0082605
1: -0.0035763, 0.0011276, -0.0035757, 0.0023879, -0.0058454, 0.0045760
2: 0.0087860, 0.0167752, 0.0066394, 0.0169192, -0.0081332, 0.0101358
3: 1.0059195, 1.0071082, 1.0058650, 1.0071342, -0.0012147, 0.0012432
4: -0.0043727, -0.0018248, -0.0043922, -0.0011136, -0.0032591, 0.0025674
5: 0.0034678, 0.0140638, 0.0035849, 0.0167398, -0.0128468, 0.0100389
6: -0.0103848, -0.0025424, -0.0126279, -0.0025368, -0.0078480, 0.0100854
7: -0.0164270, -0.0099366, -0.0174003, -0.0104205, -0.0059460, 0.0074068
8: -0.0150110, -0.0088576, -0.0151928, -0.0074051, -0.0076059, 0.0063352
9: -0.0036056, 0.0032739, -0.0054701, 0.0033438, -0.0069493, 0.0087440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 4.08 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007228, upper bound: 0.0007565
time: 0.76 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007169, upper bound: 0.0007643
time: 1.17 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005177, 0.0103950, -0.0006745, 0.0083002, -0.0082735, 0.0105303
1: -0.0035756, 0.0023879, -0.0035763, 0.0011276, -0.0045770, 0.0058454
2: 0.0066394, 0.0169192, 0.0087860, 0.0167752, -0.0101358, 0.0081332
3: 1.0058650, 1.0071342, 1.0059195, 1.0071082, -0.0012432, 0.0012147
4: -0.0043922, -0.0011136, -0.0043727, -0.0018248, -0.0025674, 0.0032591
5: 0.0035850, 0.0167398, 0.0034678, 0.0140638, -0.0100492, 0.0128468
6: -0.0126279, -0.0025368, -0.0103848, -0.0025424, -0.0100854, 0.0078480
7: -0.0174003, -0.0104209, -0.0164270, -0.0099366, -0.0074068, 0.0059473
8: -0.0151928, -0.0074051, -0.0150110, -0.0088576, -0.0063352, 0.0076059
9: -0.0054701, 0.0033438, -0.0036056, 0.0032739, -0.0087440, 0.0069493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 4.16 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007111, upper bound: 0.0007191
time: 1.06 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.97 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006745, 0.0083002, -0.0006745, 0.0083002, -0.0084340, 0.0084340
1: -0.0035763, 0.0011276, -0.0035763, 0.0011276, -0.0045817, 0.0045817
2: 0.0087860, 0.0167752, 0.0087860, 0.0167752, -0.0079893, 0.0079893
3: 1.0059195, 1.0071082, 1.0059195, 1.0071082, -0.0011888, 0.0011888
4: -0.0043727, -0.0018248, -0.0043727, -0.0018248, -0.0025479, 0.0025479
5: 0.0034678, 0.0140638, 0.0034678, 0.0140638, -0.0101692, 0.0101692
6: -0.0103848, -0.0025424, -0.0103848, -0.0025424, -0.0078424, 0.0078424
7: -0.0164270, -0.0099366, -0.0164270, -0.0099366, -0.0064336, 0.0064336
8: -0.0150110, -0.0088576, -0.0150110, -0.0088576, -0.0061535, 0.0061535
9: -0.0036056, 0.0032739, -0.0036056, 0.0032739, -0.0068795, 0.0068795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 4.07 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007111, upper bound: 0.0007193
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.99 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 6.99
Output dim: 3, lower bound: -0.0007162, upper bound: 0.0007652
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 6.99
Output dim: 3, lower bound: -0.0007169, upper bound: 0.0007643
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 6.99
Output dim: 3, lower bound: -0.0007228, upper bound: 0.0007565
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 6.99
Output dim: 3, lower bound: -0.0007169, upper bound: 0.0007643
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 6.99
Output dim: 3, lower bound: -0.0007111, upper bound: 0.0007191
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 6.99
Output dim: 3, lower bound: -0.0007131, upper bound: 0.0007131
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 6.99
Output dim: 3, lower bound: -0.0007111, upper bound: 0.0007193
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 6.99
Output dim: 3, lower bound: -0.0007131, upper bound: 0.0007131

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0005172, 0.0087532, -0.0005177, 0.0099415, -0.0098839, 0.0087022
1: -0.0035757, 0.0013898, -0.0035757, 0.0021119, -0.0055538, 0.0048355
2: 0.0082893, 0.0167834, 0.0070949, 0.0168820, -0.0085927, 0.0096885
3: 1.0059873, 1.0071342, 1.0058982, 1.0071342, -0.0011469, 0.0012360
4: -0.0043712, -0.0016617, -0.0043865, -0.0012648, -0.0031064, 0.0027247
5: 0.0035854, 0.0146464, 0.0035850, 0.0161619, -0.0121232, 0.0106131
6: -0.0108744, -0.0025394, -0.0121439, -0.0025375, -0.0083369, 0.0096044
7: -0.0166883, -0.0104205, -0.0172033, -0.0104205, -0.0062044, 0.0067190
8: -0.0149728, -0.0085120, -0.0151330, -0.0077119, -0.0072609, 0.0066210
9: -0.0040546, 0.0032390, -0.0050795, 0.0033156, -0.0073701, 0.0083185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
time: 0.83 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008342
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0006974, 0.0085953, -0.0005175, 0.0095680, -0.0096948, 0.0085571
1: -0.0036084, 0.0012982, -0.0035757, 0.0018869, -0.0053449, 0.0047478
2: 0.0084613, 0.0167772, 0.0074779, 0.0168536, -0.0083923, 0.0092994
3: 1.0059916, 1.0071584, 1.0059288, 1.0071342, -0.0011426, 0.0012296
4: -0.0043705, -0.0017180, -0.0043824, -0.0013915, -0.0029790, 0.0026644
5: 0.0034470, 0.0144438, 0.0035851, 0.0156847, -0.0117878, 0.0104209
6: -0.0107042, -0.0025408, -0.0117439, -0.0025386, -0.0081657, 0.0092031
7: -0.0165982, -0.0101471, -0.0170287, -0.0104205, -0.0061161, 0.0068183
8: -0.0149665, -0.0086243, -0.0150911, -0.0079757, -0.0069908, 0.0064668
9: -0.0038994, 0.0032361, -0.0047455, 0.0032948, -0.0071942, 0.0079817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008197
time: 0.81 seconds

## Relational analysis of IS_B1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
time: 0.94 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006742, 0.0078525, -0.0005172, 0.0087532, -0.0088973, 0.0078124
1: -0.0035763, 0.0008563, -0.0035757, 0.0013898, -0.0048395, 0.0042916
2: 0.0092382, 0.0167391, 0.0082893, 0.0167834, -0.0075452, 0.0084498
3: 1.0059503, 1.0071082, 1.0059873, 1.0071342, -0.0011839, 0.0011209
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023963
5: 0.0034680, 0.0134925, 0.0035854, 0.0146464, -0.0107599, 0.0094671
6: -0.0099060, -0.0025431, -0.0108744, -0.0025394, -0.0073666, 0.0083313
7: -0.0162294, -0.0099366, -0.0166883, -0.0104205, -0.0057484, 0.0066948
8: -0.0149536, -0.0091400, -0.0149728, -0.0085120, -0.0064416, 0.0058328
9: -0.0032162, 0.0032465, -0.0040546, 0.0032390, -0.0064552, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.79 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.84 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006740, 0.0074979, -0.0006974, 0.0085953, -0.0087543, 0.0076456
1: -0.0035763, 0.0006425, -0.0036084, 0.0012982, -0.0047524, 0.0040952
2: 0.0095979, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059789, 1.0071082, 1.0059916, 1.0071584, -0.0011795, 0.0011166
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034682, 0.0130401, 0.0034470, 0.0144438, -0.0105696, 0.0091596
6: -0.0095271, -0.0025443, -0.0107042, -0.0025408, -0.0069863, 0.0081599
7: -0.0160676, -0.0099366, -0.0165982, -0.0101471, -0.0058599, 0.0066068
8: -0.0149102, -0.0093690, -0.0149665, -0.0086243, -0.0062859, 0.0055975
9: -0.0029038, 0.0032264, -0.0038994, 0.0032361, -0.0061400, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
time: 0.98 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 1.06 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0005171, 0.0087532, -0.0006742, 0.0078525, -0.0078251, 0.0088973
1: -0.0035756, 0.0013898, -0.0035763, 0.0008563, -0.0042927, 0.0048395
2: 0.0082893, 0.0167834, 0.0092382, 0.0167391, -0.0084498, 0.0075452
3: 1.0059874, 1.0071342, 1.0059503, 1.0071082, -0.0011208, 0.0011839
4: -0.0043712, -0.0016617, -0.0043671, -0.0019749, -0.0023963, 0.0027054
5: 0.0035855, 0.0146464, 0.0034680, 0.0134925, -0.0094771, 0.0107599
6: -0.0108744, -0.0025394, -0.0099060, -0.0025431, -0.0083313, 0.0073666
7: -0.0166883, -0.0104209, -0.0162294, -0.0099366, -0.0066948, 0.0057496
8: -0.0149728, -0.0085120, -0.0149536, -0.0091400, -0.0058328, 0.0064416
9: -0.0040545, 0.0032390, -0.0032162, 0.0032465, -0.0073010, 0.0064552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
time: 0.72 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 0.92 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0006973, 0.0085953, -0.0006740, 0.0074979, -0.0076620, 0.0087543
1: -0.0036084, 0.0012982, -0.0035763, 0.0006425, -0.0040994, 0.0047524
2: 0.0084613, 0.0167772, 0.0095979, 0.0167132, -0.0082519, 0.0071793
3: 1.0059916, 1.0071584, 1.0059789, 1.0071082, -0.0011166, 0.0011795
4: -0.0043705, -0.0017180, -0.0043631, -0.0020940, -0.0022765, 0.0026451
5: 0.0034472, 0.0144438, 0.0034682, 0.0130401, -0.0091726, 0.0105696
6: -0.0107042, -0.0025408, -0.0095271, -0.0025443, -0.0081599, 0.0069863
7: -0.0165983, -0.0101474, -0.0160676, -0.0099366, -0.0066068, 0.0058613
8: -0.0149665, -0.0086244, -0.0149102, -0.0093690, -0.0055975, 0.0062859
9: -0.0038994, 0.0032361, -0.0029038, 0.0032264, -0.0071259, 0.0061400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
time: 0.72 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
time: 0.81 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0006736, 0.0066133, -0.0006743, 0.0078496, -0.0079833, 0.0067537
1: -0.0035763, 0.0001045, -0.0035763, 0.0008546, -0.0042960, 0.0035486
2: 0.0104846, 0.0166393, 0.0092411, 0.0167389, -0.0062542, 0.0073982
3: 1.0060353, 1.0071082, 1.0059507, 1.0071082, -0.0010729, 0.0011575
4: -0.0043517, -0.0023889, -0.0043671, -0.0019759, -0.0023759, 0.0019782
5: 0.0034685, 0.0119122, 0.0034680, 0.0134888, -0.0095936, 0.0080227
6: -0.0085823, -0.0025450, -0.0099030, -0.0025431, -0.0060392, 0.0073581
7: -0.0156887, -0.0099366, -0.0162282, -0.0099366, -0.0056957, 0.0062347
8: -0.0147929, -0.0098611, -0.0149532, -0.0091418, -0.0056511, 0.0050921
9: -0.0021460, 0.0031706, -0.0032137, 0.0032463, -0.0053923, 0.0063843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_B2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006660, upper bound: 0.0006571
time: 0.96 seconds

## Relational analysis of IS_B2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006475, upper bound: 0.0006555
time: 0.87 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0008555, 0.0064963, -0.0006740, 0.0074799, -0.0077934, 0.0066531
1: -0.0036091, 0.0000352, -0.0035763, 0.0006314, -0.0040932, 0.0034860
2: 0.0106145, 0.0166320, 0.0096163, 0.0167117, -0.0060973, 0.0070157
3: 1.0060413, 1.0071328, 1.0059803, 1.0071082, -0.0010669, 0.0011525
4: -0.0043508, -0.0024310, -0.0043628, -0.0021001, -0.0022507, 0.0019319
5: 0.0033290, 0.0117622, 0.0034682, 0.0130171, -0.0092611, 0.0078858
6: -0.0084564, -0.0025468, -0.0095078, -0.0025444, -0.0059120, 0.0069611
7: -0.0156142, -0.0096544, -0.0160594, -0.0099366, -0.0056230, 0.0063457
8: -0.0147852, -0.0097953, -0.0149079, -0.0093800, -0.0054051, 0.0051126
9: -0.0020268, 0.0031681, -0.0028879, 0.0032253, -0.0052522, 0.0060561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006484
time: 0.75 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006463
time: 0.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.95 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008342
IS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008197
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0006660, upper bound: 0.0006571
IS_B2_A2_A1_A2, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0006475, upper bound: 0.0006555
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006484
IS_B2_A2_A2_A2, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006463

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005172, 0.0087532, -0.0005172, 0.0087532, -0.0087020, 0.0087020
1: -0.0035757, 0.0013898, -0.0035757, 0.0013898, -0.0048279, 0.0048279
2: 0.0082893, 0.0167834, 0.0082893, 0.0167834, -0.0084941, 0.0084941
3: 1.0059873, 1.0071342, 1.0059873, 1.0071342, -0.0011469, 0.0011469
4: -0.0043712, -0.0016617, -0.0043712, -0.0016617, -0.0027094, 0.0027094
5: 0.0035854, 0.0146464, 0.0035854, 0.0146464, -0.0106129, 0.0106129
6: -0.0108744, -0.0025394, -0.0108744, -0.0025394, -0.0083350, 0.0083350
7: -0.0166883, -0.0104205, -0.0166883, -0.0104205, -0.0062045, 0.0062044
8: -0.0149728, -0.0085120, -0.0149728, -0.0085120, -0.0064608, 0.0064608
9: -0.0040546, 0.0032390, -0.0040546, 0.0032390, -0.0072936, 0.0072936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008203
time: 0.92 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
time: 0.82 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005172, 0.0087532, -0.0006974, 0.0085953, -0.0085496, 0.0088861
1: -0.0035757, 0.0013898, -0.0036084, 0.0012982, -0.0047384, 0.0048463
2: 0.0082893, 0.0167834, 0.0084613, 0.0167772, -0.0084879, 0.0083221
3: 1.0059873, 1.0071342, 1.0059916, 1.0071584, -0.0011711, 0.0011426
4: -0.0043712, -0.0016617, -0.0043705, -0.0017180, -0.0026532, 0.0027088
5: 0.0035854, 0.0146464, 0.0034470, 0.0144438, -0.0104139, 0.0107544
6: -0.0108744, -0.0025394, -0.0107042, -0.0025408, -0.0083337, 0.0081648
7: -0.0166883, -0.0104205, -0.0165982, -0.0101471, -0.0064779, 0.0061153
8: -0.0149728, -0.0085120, -0.0149665, -0.0086243, -0.0063484, 0.0064546
9: -0.0040546, 0.0032390, -0.0038994, 0.0032361, -0.0072907, 0.0071384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 1.07 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0008174
time: 0.87 seconds

## BFS IS instance: IS_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0005083, 0.0103949, -0.0104904, 0.0085464
1: -0.0036000, 0.0012982, -0.0035726, 0.0023879, -0.0058396, 0.0047476
2: 0.0084613, 0.0167772, 0.0066395, 0.0169192, -0.0084578, 0.0101378
3: 1.0059929, 1.0071430, 1.0058657, 1.0071290, -0.0011361, 0.0012773
4: -0.0043705, -0.0017180, -0.0043922, -0.0011136, -0.0032569, 0.0026742
5: 0.0034685, 0.0144438, 0.0035923, 0.0167398, -0.0128187, 0.0104125
6: -0.0107042, -0.0025416, -0.0126279, -0.0025370, -0.0081672, 0.0100863
7: -0.0165982, -0.0101711, -0.0174003, -0.0104289, -0.0061076, 0.0071655
8: -0.0149665, -0.0086314, -0.0151928, -0.0074054, -0.0075611, 0.0065614
9: -0.0038993, 0.0032361, -0.0054700, 0.0033438, -0.0072431, 0.0087062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007989
time: 0.92 seconds

## BFS IS instance: IS_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0005032, 0.0103949, -0.0104894, 0.0088965
1: -0.0035939, 0.0015108, -0.0035697, 0.0023879, -0.0058408, 0.0049626
2: 0.0081186, 0.0168105, 0.0066395, 0.0169192, -0.0088006, 0.0101710
3: 1.0059661, 1.0071237, 1.0058658, 1.0071189, -0.0011529, 0.0012579
4: -0.0043762, -0.0016037, -0.0043922, -0.0011136, -0.0032626, 0.0027885
5: 0.0034765, 0.0148873, 0.0035964, 0.0167398, -0.0128179, 0.0108575
6: -0.0110755, -0.0025432, -0.0126279, -0.0025378, -0.0085377, 0.0100847
7: -0.0167340, -0.0101752, -0.0174003, -0.0104316, -0.0062419, 0.0071625
8: -0.0150302, -0.0084228, -0.0151928, -0.0074056, -0.0076246, 0.0067699
9: -0.0041862, 0.0032678, -0.0054700, 0.0033438, -0.0075299, 0.0087379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
time: 0.97 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007986
time: 0.89 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0005080, 0.0087532, -0.0088649, 0.0082496
1: -0.0035670, 0.0011276, -0.0035728, 0.0013898, -0.0048295, 0.0045619
2: 0.0087861, 0.0167752, 0.0082893, 0.0167834, -0.0079973, 0.0084859
3: 1.0059211, 1.0070909, 1.0059881, 1.0071293, -0.0012082, 0.0011028
4: -0.0043727, -0.0018248, -0.0043712, -0.0016617, -0.0027110, 0.0025464
5: 0.0034913, 0.0140638, 0.0035925, 0.0146464, -0.0107347, 0.0100305
6: -0.0103848, -0.0025434, -0.0108744, -0.0025397, -0.0078451, 0.0083311
7: -0.0164270, -0.0099621, -0.0166883, -0.0104285, -0.0059379, 0.0066694
8: -0.0150110, -0.0088693, -0.0149728, -0.0085126, -0.0064985, 0.0061034
9: -0.0036054, 0.0032739, -0.0040545, 0.0032390, -0.0068444, 0.0073284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.81 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.88 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0005046, 0.0087532, -0.0088676, 0.0085686
1: -0.0035629, 0.0013207, -0.0035705, 0.0013898, -0.0048307, 0.0047573
2: 0.0084764, 0.0168061, 0.0082893, 0.0167834, -0.0083071, 0.0085168
3: 1.0058957, 1.0070760, 1.0059879, 1.0071208, -0.0012251, 0.0010881
4: -0.0043781, -0.0017212, -0.0043712, -0.0016617, -0.0027164, 0.0026500
5: 0.0034944, 0.0144650, 0.0035953, 0.0146464, -0.0107368, 0.0104353
6: -0.0107207, -0.0025446, -0.0108744, -0.0025404, -0.0081803, 0.0083298
7: -0.0165489, -0.0099628, -0.0166883, -0.0104301, -0.0060587, 0.0066686
8: -0.0150734, -0.0086872, -0.0149728, -0.0085127, -0.0065606, 0.0062856
9: -0.0038642, 0.0033052, -0.0040545, 0.0032390, -0.0071032, 0.0073597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.85 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.95 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0006888, 0.0085953, -0.0087219, 0.0084352
1: -0.0035670, 0.0011276, -0.0036057, 0.0012982, -0.0047433, 0.0045807
2: 0.0087861, 0.0167752, 0.0084613, 0.0167772, -0.0079911, 0.0083139
3: 1.0059211, 1.0070909, 1.0059921, 1.0071536, -0.0012325, 0.0010989
4: -0.0043727, -0.0018248, -0.0043705, -0.0017180, -0.0026547, 0.0025457
5: 0.0034913, 0.0140638, 0.0034537, 0.0144437, -0.0105444, 0.0101732
6: -0.0103848, -0.0025434, -0.0107042, -0.0025410, -0.0078437, 0.0081609
7: -0.0164270, -0.0099621, -0.0165982, -0.0101546, -0.0062118, 0.0065814
8: -0.0150110, -0.0088693, -0.0149665, -0.0086267, -0.0063844, 0.0060972
9: -0.0036054, 0.0032739, -0.0038994, 0.0032361, -0.0068415, 0.0071733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.87 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
time: 0.96 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0006834, 0.0085953, -0.0087227, 0.0087492
1: -0.0035629, 0.0013207, -0.0036029, 0.0012982, -0.0047436, 0.0047749
2: 0.0084764, 0.0168061, 0.0084613, 0.0167772, -0.0083009, 0.0083448
3: 1.0058957, 1.0070760, 1.0059923, 1.0071450, -0.0012493, 0.0010837
4: -0.0043781, -0.0017212, -0.0043705, -0.0017180, -0.0026601, 0.0026493
5: 0.0034944, 0.0144650, 0.0034580, 0.0144438, -0.0105450, 0.0105741
6: -0.0107207, -0.0025446, -0.0107043, -0.0025417, -0.0081790, 0.0081596
7: -0.0165489, -0.0099628, -0.0165982, -0.0101576, -0.0063314, 0.0065803
8: -0.0150734, -0.0086872, -0.0149665, -0.0086297, -0.0064437, 0.0062793
9: -0.0038642, 0.0033052, -0.0038994, 0.0032361, -0.0071004, 0.0072046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.90 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 0.95 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005079, 0.0087532, -0.0006444, 0.0083002, -0.0082624, 0.0088649
1: -0.0035727, 0.0013898, -0.0035670, 0.0011276, -0.0045629, 0.0048295
2: 0.0082893, 0.0167834, 0.0087861, 0.0167752, -0.0084859, 0.0079973
3: 1.0059880, 1.0071293, 1.0059211, 1.0070909, -0.0011029, 0.0012082
4: -0.0043712, -0.0016617, -0.0043727, -0.0018248, -0.0025464, 0.0027110
5: 0.0035926, 0.0146464, 0.0034913, 0.0140638, -0.0100405, 0.0107347
6: -0.0108744, -0.0025397, -0.0103848, -0.0025434, -0.0083311, 0.0078451
7: -0.0166883, -0.0104289, -0.0164270, -0.0099621, -0.0066694, 0.0059391
8: -0.0149728, -0.0085126, -0.0150110, -0.0088693, -0.0061034, 0.0064984
9: -0.0040545, 0.0032390, -0.0036054, 0.0032739, -0.0073284, 0.0068444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
time: 0.72 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.80 seconds

## BFS IS instance: IS_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005044, 0.0087532, -0.0006403, 0.0086152, -0.0085822, 0.0088676
1: -0.0035704, 0.0013898, -0.0035629, 0.0013207, -0.0047589, 0.0048307
2: 0.0082893, 0.0167834, 0.0084764, 0.0168061, -0.0085168, 0.0083071
3: 1.0059880, 1.0071208, 1.0058957, 1.0070760, -0.0010880, 0.0012251
4: -0.0043712, -0.0016617, -0.0043781, -0.0017212, -0.0026500, 0.0027164
5: 0.0035954, 0.0146464, 0.0034944, 0.0144650, -0.0104461, 0.0107368
6: -0.0108744, -0.0025404, -0.0107207, -0.0025446, -0.0083298, 0.0081803
7: -0.0166883, -0.0104305, -0.0165489, -0.0099628, -0.0066686, 0.0060600
8: -0.0149728, -0.0085128, -0.0150734, -0.0086872, -0.0062856, 0.0065606
9: -0.0040545, 0.0032390, -0.0038642, 0.0033052, -0.0073597, 0.0071032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 0.95 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006933
time: 0.99 seconds

## BFS IS instance: IS_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006886, 0.0085954, -0.0006444, 0.0083002, -0.0084520, 0.0087219
1: -0.0036057, 0.0012982, -0.0035670, 0.0011276, -0.0045846, 0.0047433
2: 0.0084613, 0.0167772, 0.0087861, 0.0167752, -0.0083139, 0.0079911
3: 1.0059922, 1.0071536, 1.0059211, 1.0070909, -0.0010988, 0.0012325
4: -0.0043705, -0.0017180, -0.0043727, -0.0018248, -0.0025457, 0.0026547
5: 0.0034539, 0.0144438, 0.0034913, 0.0140638, -0.0101864, 0.0105444
6: -0.0107042, -0.0025410, -0.0103848, -0.0025434, -0.0081609, 0.0078437
7: -0.0165982, -0.0101550, -0.0164270, -0.0099621, -0.0065814, 0.0062132
8: -0.0149665, -0.0086267, -0.0150110, -0.0088693, -0.0060972, 0.0063843
9: -0.0038994, 0.0032361, -0.0036054, 0.0032739, -0.0071733, 0.0068415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.80 seconds

## Relational analysis of IS_B2_A1_A2_B1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006832, 0.0085953, -0.0006403, 0.0086152, -0.0087667, 0.0087227
1: -0.0036029, 0.0012982, -0.0035629, 0.0013207, -0.0047794, 0.0047436
2: 0.0084613, 0.0167772, 0.0084764, 0.0168061, -0.0083448, 0.0083009
3: 1.0059924, 1.0071450, 1.0058957, 1.0070760, -0.0010836, 0.0012493
4: -0.0043705, -0.0017180, -0.0043781, -0.0017212, -0.0026493, 0.0026601
5: 0.0034581, 0.0144438, 0.0034944, 0.0144650, -0.0105878, 0.0105450
6: -0.0107042, -0.0025417, -0.0107207, -0.0025446, -0.0081596, 0.0081790
7: -0.0165982, -0.0101579, -0.0165489, -0.0099628, -0.0065803, 0.0063328
8: -0.0149665, -0.0086297, -0.0150734, -0.0086872, -0.0062793, 0.0064437
9: -0.0038994, 0.0032361, -0.0038642, 0.0033052, -0.0072046, 0.0071004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006933
time: 0.97 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0006631, 0.0066133, -0.0006720, 0.0083003, -0.0084129, 0.0067432
1: -0.0035763, 0.0001045, -0.0035763, 0.0011276, -0.0045713, 0.0035483
2: 0.0104847, 0.0166393, 0.0087860, 0.0167752, -0.0062906, 0.0078533
3: 1.0060581, 1.0071082, 1.0059242, 1.0071082, -0.0010501, 0.0011840
4: -0.0043514, -0.0023889, -0.0043726, -0.0018248, -0.0025266, 0.0019838
5: 0.0034759, 0.0119122, 0.0034695, 0.0140638, -0.0101535, 0.0080146
6: -0.0085823, -0.0025450, -0.0103848, -0.0025424, -0.0060399, 0.0078398
7: -0.0156886, -0.0099997, -0.0164270, -0.0099514, -0.0056801, 0.0063691
8: -0.0147810, -0.0098703, -0.0150084, -0.0088583, -0.0059227, 0.0051381
9: -0.0021458, 0.0031600, -0.0036055, 0.0032716, -0.0054174, 0.0067655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006660, upper bound: 0.0006571
time: 0.83 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006660, upper bound: 0.0006571
time: 0.91 seconds

## BFS IS instance: IS_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0008449, 0.0064963, -0.0006720, 0.0083003, -0.0085897, 0.0066428
1: -0.0036091, 0.0000352, -0.0035763, 0.0011276, -0.0045918, 0.0034869
2: 0.0106146, 0.0166320, 0.0087860, 0.0167752, -0.0061607, 0.0078460
3: 1.0060627, 1.0071328, 1.0059242, 1.0071082, -0.0010455, 0.0012085
4: -0.0043504, -0.0024310, -0.0043726, -0.0018248, -0.0025257, 0.0019416
5: 0.0033365, 0.0117622, 0.0034695, 0.0140638, -0.0102896, 0.0078778
6: -0.0084564, -0.0025468, -0.0103848, -0.0025424, -0.0059139, 0.0078380
7: -0.0156142, -0.0097188, -0.0164270, -0.0099514, -0.0056073, 0.0066484
8: -0.0147738, -0.0098101, -0.0150084, -0.0088583, -0.0059154, 0.0051983
9: -0.0020266, 0.0031579, -0.0036055, 0.0032716, -0.0052982, 0.0067635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006472
time: 0.97 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006472
time: 0.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.38 seconds
IS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008203
IS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0008174
IS_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007989
IS_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
IS_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007986
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006933
IS_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
IS_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006933
IS_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
IS_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006660, upper bound: 0.0006571
IS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006660, upper bound: 0.0006571
IS_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006472
IS_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006472

## BFS IS instance: IS_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005172, 0.0087532, -0.0005172, 0.0087532, -0.0087020, 0.0087020
1: -0.0035757, 0.0013898, -0.0035757, 0.0013898, -0.0048279, 0.0048279
2: 0.0082893, 0.0167834, 0.0082893, 0.0167834, -0.0084941, 0.0084941
3: 1.0059873, 1.0071342, 1.0059873, 1.0071342, -0.0011469, 0.0011469
4: -0.0043712, -0.0016617, -0.0043712, -0.0016617, -0.0027094, 0.0027094
5: 0.0035854, 0.0146464, 0.0035854, 0.0146464, -0.0106129, 0.0106129
6: -0.0108744, -0.0025394, -0.0108744, -0.0025394, -0.0083350, 0.0083350
7: -0.0166883, -0.0104205, -0.0166883, -0.0104205, -0.0062045, 0.0062044
8: -0.0149728, -0.0085120, -0.0149728, -0.0085120, -0.0064608, 0.0064608
9: -0.0040546, 0.0032390, -0.0040546, 0.0032390, -0.0072936, 0.0072936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008203, upper bound: 0.0008151
time: 0.93 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006974, 0.0085953, -0.0005172, 0.0087532, -0.0088861, 0.0085496
1: -0.0036084, 0.0012982, -0.0035757, 0.0013898, -0.0048463, 0.0047384
2: 0.0084613, 0.0167772, 0.0082893, 0.0167834, -0.0083221, 0.0084879
3: 1.0059916, 1.0071584, 1.0059873, 1.0071342, -0.0011426, 0.0011711
4: -0.0043705, -0.0017180, -0.0043712, -0.0016617, -0.0027088, 0.0026532
5: 0.0034470, 0.0144438, 0.0035854, 0.0146464, -0.0107544, 0.0104139
6: -0.0107042, -0.0025408, -0.0108744, -0.0025394, -0.0081648, 0.0083337
7: -0.0165982, -0.0101471, -0.0166883, -0.0104205, -0.0061153, 0.0064779
8: -0.0149665, -0.0086243, -0.0149728, -0.0085120, -0.0064546, 0.0063484
9: -0.0038994, 0.0032361, -0.0040546, 0.0032390, -0.0071384, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008024, upper bound: 0.0007982
time: 0.87 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008024, upper bound: 0.0007978
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005077, 0.0087532, -0.0006698, 0.0085953, -0.0085387, 0.0088576
1: -0.0035726, 0.0013898, -0.0036000, 0.0012982, -0.0047345, 0.0048363
2: 0.0082893, 0.0167834, 0.0084613, 0.0167772, -0.0084879, 0.0083221
3: 1.0059879, 1.0071290, 1.0059929, 1.0071430, -0.0011551, 0.0011361
4: -0.0043712, -0.0016617, -0.0043705, -0.0017180, -0.0026532, 0.0027088
5: 0.0035928, 0.0146464, 0.0034685, 0.0144438, -0.0104054, 0.0107323
6: -0.0108744, -0.0025397, -0.0107042, -0.0025416, -0.0083328, 0.0081645
7: -0.0166883, -0.0104289, -0.0165982, -0.0101711, -0.0064536, 0.0061068
8: -0.0149728, -0.0085126, -0.0149665, -0.0086314, -0.0063414, 0.0064539
9: -0.0040545, 0.0032390, -0.0038993, 0.0032361, -0.0072907, 0.0071383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0007978
time: 0.85 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005026, 0.0087532, -0.0006596, 0.0089437, -0.0088841, 0.0088565
1: -0.0035697, 0.0013898, -0.0035939, 0.0015108, -0.0049463, 0.0048375
2: 0.0082893, 0.0167834, 0.0081186, 0.0168105, -0.0085212, 0.0086649
3: 1.0059880, 1.0071189, 1.0059661, 1.0071237, -0.0011357, 0.0011529
4: -0.0043712, -0.0016617, -0.0043762, -0.0016037, -0.0027675, 0.0027145
5: 0.0035968, 0.0146464, 0.0034765, 0.0148873, -0.0108475, 0.0107316
6: -0.0108744, -0.0025405, -0.0110755, -0.0025432, -0.0083313, 0.0085350
7: -0.0166883, -0.0104316, -0.0167340, -0.0101752, -0.0064506, 0.0062403
8: -0.0149728, -0.0085129, -0.0150302, -0.0084228, -0.0065499, 0.0065173
9: -0.0040545, 0.0032390, -0.0041862, 0.0032678, -0.0073224, 0.0074252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0008174
time: 0.86 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007978
time: 0.84 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0005172, 0.0087532, -0.0088576, 0.0085496
1: -0.0036000, 0.0012982, -0.0035757, 0.0013898, -0.0048363, 0.0047383
2: 0.0084613, 0.0167772, 0.0082893, 0.0167834, -0.0083221, 0.0084879
3: 1.0059929, 1.0071430, 1.0059873, 1.0071342, -0.0011413, 0.0011557
4: -0.0043705, -0.0017180, -0.0043712, -0.0016617, -0.0027088, 0.0026532
5: 0.0034685, 0.0144438, 0.0035854, 0.0146464, -0.0107323, 0.0104139
6: -0.0107042, -0.0025416, -0.0108744, -0.0025394, -0.0081648, 0.0083328
7: -0.0165982, -0.0101711, -0.0166883, -0.0104205, -0.0061152, 0.0064536
8: -0.0149665, -0.0086314, -0.0149728, -0.0085120, -0.0064546, 0.0063414
9: -0.0038993, 0.0032361, -0.0040546, 0.0032390, -0.0071383, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 0.98 seconds

## Relational analysis of IS_B1_A1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.86 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006974, 0.0085953, -0.0085905, 0.0086219
1: -0.0036000, 0.0012982, -0.0036084, 0.0012982, -0.0047324, 0.0047441
2: 0.0084613, 0.0167772, 0.0084613, 0.0167772, -0.0083159, 0.0083159
3: 1.0059929, 1.0071430, 1.0059916, 1.0071584, -0.0011655, 0.0011514
4: -0.0043705, -0.0017180, -0.0043705, -0.0017180, -0.0026525, 0.0026525
5: 0.0034685, 0.0144438, 0.0034470, 0.0144438, -0.0104438, 0.0104683
6: -0.0107042, -0.0025416, -0.0107042, -0.0025408, -0.0081634, 0.0081626
7: -0.0165982, -0.0101711, -0.0165982, -0.0101471, -0.0063771, 0.0063526
8: -0.0149665, -0.0086314, -0.0149665, -0.0086243, -0.0063422, 0.0063351
9: -0.0038993, 0.0032361, -0.0038994, 0.0032361, -0.0071355, 0.0071356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008197
time: 0.80 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008197
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0005172, 0.0087532, -0.0088565, 0.0088956
1: -0.0035939, 0.0015108, -0.0035757, 0.0013898, -0.0048375, 0.0049515
2: 0.0081186, 0.0168105, 0.0082893, 0.0167834, -0.0086649, 0.0085212
3: 1.0059661, 1.0071237, 1.0059873, 1.0071342, -0.0011681, 0.0011364
4: -0.0043762, -0.0016037, -0.0043712, -0.0016617, -0.0027145, 0.0027675
5: 0.0034765, 0.0148873, 0.0035854, 0.0146464, -0.0107315, 0.0108564
6: -0.0110755, -0.0025432, -0.0108744, -0.0025394, -0.0085361, 0.0083313
7: -0.0167340, -0.0101752, -0.0166883, -0.0104205, -0.0062512, 0.0064506
8: -0.0150302, -0.0084228, -0.0149728, -0.0085120, -0.0065182, 0.0065499
9: -0.0041862, 0.0032678, -0.0040546, 0.0032390, -0.0074252, 0.0073224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
time: 0.93 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006974, 0.0085953, -0.0085887, 0.0089725
1: -0.0035939, 0.0015108, -0.0036084, 0.0012982, -0.0047321, 0.0049605
2: 0.0081186, 0.0168105, 0.0084613, 0.0167772, -0.0086587, 0.0083492
3: 1.0059661, 1.0071237, 1.0059916, 1.0071584, -0.0011923, 0.0011321
4: -0.0043762, -0.0016037, -0.0043705, -0.0017180, -0.0026582, 0.0027668
5: 0.0034765, 0.0148873, 0.0034470, 0.0144438, -0.0104425, 0.0109135
6: -0.0110755, -0.0025432, -0.0107042, -0.0025408, -0.0085347, 0.0081611
7: -0.0167340, -0.0101752, -0.0165982, -0.0101471, -0.0065138, 0.0063497
8: -0.0150302, -0.0084228, -0.0149665, -0.0086243, -0.0064058, 0.0065437
9: -0.0041862, 0.0032678, -0.0038994, 0.0032361, -0.0074223, 0.0071673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
time: 0.91 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.81 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.78 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.91 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.02 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006401, 0.0081766, -0.0005172, 0.0087532, -0.0088675, 0.0081392
1: -0.0035629, 0.0010552, -0.0035757, 0.0013898, -0.0048281, 0.0044937
2: 0.0089210, 0.0167706, 0.0082893, 0.0167834, -0.0078624, 0.0084813
3: 1.0059268, 1.0070760, 1.0059873, 1.0071342, -0.0012074, 0.0010887
4: -0.0043726, -0.0018687, -0.0043712, -0.0016617, -0.0027109, 0.0025025
5: 0.0034946, 0.0139053, 0.0035854, 0.0146464, -0.0107368, 0.0098828
6: -0.0102517, -0.0025453, -0.0108744, -0.0025394, -0.0077123, 0.0083292
7: -0.0163553, -0.0099628, -0.0166883, -0.0104205, -0.0058753, 0.0066686
8: -0.0150165, -0.0089727, -0.0149728, -0.0085120, -0.0065045, 0.0060001
9: -0.0034833, 0.0032784, -0.0040546, 0.0032390, -0.0067223, 0.0073330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.78 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.84 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006399, 0.0077967, -0.0006974, 0.0085953, -0.0087225, 0.0079487
1: -0.0035629, 0.0008257, -0.0036084, 0.0012982, -0.0047399, 0.0042809
2: 0.0093030, 0.0167430, 0.0084613, 0.0167772, -0.0074742, 0.0082816
3: 1.0059569, 1.0070760, 1.0059916, 1.0071584, -0.0012015, 0.0010844
4: -0.0043683, -0.0019954, -0.0043705, -0.0017180, -0.0026503, 0.0023751
5: 0.0034948, 0.0134210, 0.0034470, 0.0144438, -0.0105449, 0.0095437
6: -0.0098461, -0.0025465, -0.0107042, -0.0025408, -0.0073053, 0.0081578
7: -0.0161838, -0.0099628, -0.0165982, -0.0101471, -0.0059765, 0.0065804
8: -0.0149672, -0.0092253, -0.0149665, -0.0086243, -0.0063429, 0.0057412
9: -0.0031504, 0.0032561, -0.0038994, 0.0032361, -0.0063865, 0.0071556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.05 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.85 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.80 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
time: 1.00 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 1.01 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006401, 0.0081766, -0.0005172, 0.0087532, -0.0088675, 0.0081392
1: -0.0035629, 0.0010552, -0.0035757, 0.0013898, -0.0048281, 0.0044937
2: 0.0089210, 0.0167706, 0.0082893, 0.0167834, -0.0078624, 0.0084813
3: 1.0059268, 1.0070760, 1.0059873, 1.0071342, -0.0012074, 0.0010887
4: -0.0043726, -0.0018687, -0.0043712, -0.0016617, -0.0027109, 0.0025025
5: 0.0034946, 0.0139053, 0.0035854, 0.0146464, -0.0107368, 0.0098828
6: -0.0102517, -0.0025453, -0.0108744, -0.0025394, -0.0077123, 0.0083292
7: -0.0163553, -0.0099628, -0.0166883, -0.0104205, -0.0058753, 0.0066686
8: -0.0150165, -0.0089727, -0.0149728, -0.0085120, -0.0065045, 0.0060001
9: -0.0034833, 0.0032784, -0.0040546, 0.0032390, -0.0067223, 0.0073330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.81 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.81 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006399, 0.0077967, -0.0006974, 0.0085953, -0.0087225, 0.0079487
1: -0.0035629, 0.0008257, -0.0036084, 0.0012982, -0.0047399, 0.0042809
2: 0.0093030, 0.0167430, 0.0084613, 0.0167772, -0.0074742, 0.0082816
3: 1.0059569, 1.0070760, 1.0059916, 1.0071584, -0.0012015, 0.0010844
4: -0.0043683, -0.0019954, -0.0043705, -0.0017180, -0.0026503, 0.0023751
5: 0.0034948, 0.0134210, 0.0034470, 0.0144438, -0.0105449, 0.0095437
6: -0.0098461, -0.0025465, -0.0107042, -0.0025408, -0.0073053, 0.0081578
7: -0.0161838, -0.0099628, -0.0165982, -0.0101471, -0.0059765, 0.0065804
8: -0.0149672, -0.0092253, -0.0149665, -0.0086243, -0.0063429, 0.0057412
9: -0.0031504, 0.0032561, -0.0038994, 0.0032361, -0.0063865, 0.0071556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 1.02 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 0.99 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005171, 0.0087532, -0.0006442, 0.0078524, -0.0078250, 0.0088648
1: -0.0035756, 0.0013898, -0.0035670, 0.0008563, -0.0042927, 0.0048269
2: 0.0082893, 0.0167834, 0.0092383, 0.0167391, -0.0084498, 0.0075451
3: 1.0059874, 1.0071342, 1.0059520, 1.0070909, -0.0011035, 0.0011822
4: -0.0043712, -0.0016617, -0.0043671, -0.0019749, -0.0023962, 0.0027054
5: 0.0035855, 0.0146464, 0.0034914, 0.0134924, -0.0094771, 0.0107346
6: -0.0108744, -0.0025394, -0.0099061, -0.0025440, -0.0083304, 0.0073666
7: -0.0166883, -0.0104209, -0.0162294, -0.0099621, -0.0066694, 0.0057496
8: -0.0149728, -0.0085120, -0.0149536, -0.0091554, -0.0058174, 0.0064416
9: -0.0040545, 0.0032390, -0.0032160, 0.0032465, -0.0073010, 0.0064550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
time: 0.73 seconds

## Relational analysis of IS_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 1.06 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006973, 0.0085953, -0.0006439, 0.0074979, -0.0076620, 0.0087218
1: -0.0036084, 0.0012982, -0.0035670, 0.0006425, -0.0040994, 0.0047397
2: 0.0084613, 0.0167772, 0.0095980, 0.0167132, -0.0082519, 0.0071793
3: 1.0059916, 1.0071584, 1.0059806, 1.0070909, -0.0010993, 0.0011778
4: -0.0043705, -0.0017180, -0.0043631, -0.0020940, -0.0022765, 0.0026451
5: 0.0034472, 0.0144438, 0.0034916, 0.0130401, -0.0091726, 0.0105442
6: -0.0107042, -0.0025408, -0.0095271, -0.0025453, -0.0081590, 0.0069863
7: -0.0165983, -0.0101474, -0.0160676, -0.0099622, -0.0065814, 0.0058612
8: -0.0149665, -0.0086244, -0.0149102, -0.0093882, -0.0055783, 0.0062859
9: -0.0038994, 0.0032361, -0.0029036, 0.0032264, -0.0071259, 0.0061397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.77 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 0.82 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005171, 0.0087532, -0.0006401, 0.0081766, -0.0081519, 0.0088675
1: -0.0035756, 0.0013898, -0.0035629, 0.0010552, -0.0044948, 0.0048281
2: 0.0082893, 0.0167834, 0.0089210, 0.0167706, -0.0084813, 0.0078624
3: 1.0059874, 1.0071342, 1.0059268, 1.0070760, -0.0010886, 0.0012074
4: -0.0043712, -0.0016617, -0.0043726, -0.0018687, -0.0025025, 0.0027109
5: 0.0035855, 0.0146464, 0.0034946, 0.0139053, -0.0098928, 0.0107368
6: -0.0108744, -0.0025394, -0.0102517, -0.0025453, -0.0083291, 0.0077123
7: -0.0166883, -0.0104209, -0.0163553, -0.0099628, -0.0066686, 0.0058765
8: -0.0149728, -0.0085120, -0.0150165, -0.0089727, -0.0060001, 0.0065045
9: -0.0040545, 0.0032390, -0.0034833, 0.0032784, -0.0073329, 0.0067223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 1.06 seconds

## Relational analysis of IS_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 0.93 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006973, 0.0085953, -0.0006399, 0.0077967, -0.0079663, 0.0087226
1: -0.0036084, 0.0012982, -0.0035629, 0.0008257, -0.0042853, 0.0047399
2: 0.0084613, 0.0167772, 0.0093030, 0.0167430, -0.0082816, 0.0074742
3: 1.0059916, 1.0071584, 1.0059569, 1.0070760, -0.0010844, 0.0012015
4: -0.0043705, -0.0017180, -0.0043683, -0.0019954, -0.0023751, 0.0026503
5: 0.0034472, 0.0144438, 0.0034948, 0.0134210, -0.0095574, 0.0105449
6: -0.0107042, -0.0025408, -0.0098461, -0.0025465, -0.0081578, 0.0073053
7: -0.0165983, -0.0101474, -0.0161838, -0.0099628, -0.0065804, 0.0059780
8: -0.0149665, -0.0086244, -0.0149672, -0.0092253, -0.0057412, 0.0063429
9: -0.0038994, 0.0032361, -0.0031504, 0.0032561, -0.0071556, 0.0063865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 0.82 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 0.79 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005171, 0.0087532, -0.0006442, 0.0078524, -0.0078250, 0.0088648
1: -0.0035756, 0.0013898, -0.0035670, 0.0008563, -0.0042927, 0.0048269
2: 0.0082893, 0.0167834, 0.0092383, 0.0167391, -0.0084498, 0.0075451
3: 1.0059874, 1.0071342, 1.0059520, 1.0070909, -0.0011035, 0.0011822
4: -0.0043712, -0.0016617, -0.0043671, -0.0019749, -0.0023962, 0.0027054
5: 0.0035855, 0.0146464, 0.0034914, 0.0134924, -0.0094771, 0.0107346
6: -0.0108744, -0.0025394, -0.0099061, -0.0025440, -0.0083304, 0.0073666
7: -0.0166883, -0.0104209, -0.0162294, -0.0099621, -0.0066694, 0.0057496
8: -0.0149728, -0.0085120, -0.0149536, -0.0091554, -0.0058174, 0.0064416
9: -0.0040545, 0.0032390, -0.0032160, 0.0032465, -0.0073010, 0.0064550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 1.01 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006973, 0.0085953, -0.0006439, 0.0074979, -0.0076620, 0.0087218
1: -0.0036084, 0.0012982, -0.0035670, 0.0006425, -0.0040994, 0.0047397
2: 0.0084613, 0.0167772, 0.0095980, 0.0167132, -0.0082519, 0.0071793
3: 1.0059916, 1.0071584, 1.0059806, 1.0070909, -0.0010993, 0.0011778
4: -0.0043705, -0.0017180, -0.0043631, -0.0020940, -0.0022765, 0.0026451
5: 0.0034472, 0.0144438, 0.0034916, 0.0130401, -0.0091726, 0.0105442
6: -0.0107042, -0.0025408, -0.0095271, -0.0025453, -0.0081590, 0.0069863
7: -0.0165983, -0.0101474, -0.0160676, -0.0099622, -0.0065814, 0.0058612
8: -0.0149665, -0.0086244, -0.0149102, -0.0093882, -0.0055783, 0.0062859
9: -0.0038994, 0.0032361, -0.0029036, 0.0032264, -0.0071259, 0.0061397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
time: 0.81 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005171, 0.0087532, -0.0006401, 0.0081766, -0.0081519, 0.0088675
1: -0.0035756, 0.0013898, -0.0035629, 0.0010552, -0.0044948, 0.0048281
2: 0.0082893, 0.0167834, 0.0089210, 0.0167706, -0.0084813, 0.0078624
3: 1.0059874, 1.0071342, 1.0059268, 1.0070760, -0.0010886, 0.0012074
4: -0.0043712, -0.0016617, -0.0043726, -0.0018687, -0.0025025, 0.0027109
5: 0.0035855, 0.0146464, 0.0034946, 0.0139053, -0.0098928, 0.0107368
6: -0.0108744, -0.0025394, -0.0102517, -0.0025453, -0.0083291, 0.0077123
7: -0.0166883, -0.0104209, -0.0163553, -0.0099628, -0.0066686, 0.0058765
8: -0.0149728, -0.0085120, -0.0150165, -0.0089727, -0.0060001, 0.0065045
9: -0.0040545, 0.0032390, -0.0034833, 0.0032784, -0.0073329, 0.0067223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 0.96 seconds

## Relational analysis of IS_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 0.96 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006973, 0.0085953, -0.0006399, 0.0077967, -0.0079663, 0.0087226
1: -0.0036084, 0.0012982, -0.0035629, 0.0008257, -0.0042853, 0.0047399
2: 0.0084613, 0.0167772, 0.0093030, 0.0167430, -0.0082816, 0.0074742
3: 1.0059916, 1.0071584, 1.0059569, 1.0070760, -0.0010844, 0.0012015
4: -0.0043705, -0.0017180, -0.0043683, -0.0019954, -0.0023751, 0.0026503
5: 0.0034472, 0.0144438, 0.0034948, 0.0134210, -0.0095574, 0.0105449
6: -0.0107042, -0.0025408, -0.0098461, -0.0025465, -0.0081578, 0.0073053
7: -0.0165983, -0.0101474, -0.0161838, -0.0099628, -0.0065804, 0.0059780
8: -0.0149665, -0.0086244, -0.0149672, -0.0092253, -0.0057412, 0.0063429
9: -0.0038994, 0.0032361, -0.0031504, 0.0032561, -0.0071556, 0.0063865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006631, 0.0066133, -0.0006736, 0.0066133, -0.0067340, 0.0067474
1: -0.0035763, 0.0001045, -0.0035763, 0.0001045, -0.0035415, 0.0035386
2: 0.0104847, 0.0166393, 0.0104846, 0.0166393, -0.0061546, 0.0061547
3: 1.0060581, 1.0071082, 1.0060353, 1.0071082, -0.0010501, 0.0010729
4: -0.0043514, -0.0023889, -0.0043517, -0.0023889, -0.0019625, 0.0019629
5: 0.0034759, 0.0119122, 0.0034685, 0.0119122, -0.0080081, 0.0080177
6: -0.0085823, -0.0025450, -0.0085823, -0.0025450, -0.0060374, 0.0060374
7: -0.0156886, -0.0099997, -0.0156887, -0.0099366, -0.0056950, 0.0056314
8: -0.0147810, -0.0098703, -0.0147929, -0.0098611, -0.0049199, 0.0049226
9: -0.0021458, 0.0031600, -0.0021460, 0.0031706, -0.0053164, 0.0053060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_B2_A2_A1_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006652, upper bound: 0.0006472
time: 0.80 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006652, upper bound: 0.0006472
time: 0.83 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006631, 0.0066133, -0.0008555, 0.0064963, -0.0066208, 0.0069230
1: -0.0035763, 0.0001045, -0.0036091, 0.0000352, -0.0034749, 0.0035590
2: 0.0104847, 0.0166393, 0.0106145, 0.0166320, -0.0061473, 0.0060248
3: 1.0060581, 1.0071082, 1.0060413, 1.0071328, -0.0010747, 0.0010669
4: -0.0043514, -0.0023889, -0.0043508, -0.0024310, -0.0019204, 0.0019619
5: 0.0034759, 0.0119122, 0.0033290, 0.0117622, -0.0078607, 0.0081528
6: -0.0085823, -0.0025450, -0.0084564, -0.0025468, -0.0060356, 0.0059114
7: -0.0156886, -0.0099997, -0.0156142, -0.0096544, -0.0059750, 0.0055564
8: -0.0147810, -0.0098703, -0.0147852, -0.0097953, -0.0049857, 0.0049149
9: -0.0021458, 0.0031600, -0.0020268, 0.0031681, -0.0053140, 0.0051868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_B2_A2_A1_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006652, upper bound: 0.0006484
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006652, upper bound: 0.0006484
time: 0.79 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008449, 0.0064963, -0.0006736, 0.0066133, -0.0069108, 0.0066342
1: -0.0036091, 0.0000352, -0.0035763, 0.0001045, -0.0035619, 0.0034720
2: 0.0106146, 0.0166320, 0.0104846, 0.0166393, -0.0060247, 0.0061474
3: 1.0060627, 1.0071328, 1.0060353, 1.0071082, -0.0010455, 0.0010974
4: -0.0043504, -0.0024310, -0.0043517, -0.0023889, -0.0019616, 0.0019207
5: 0.0033365, 0.0117622, 0.0034685, 0.0119122, -0.0081441, 0.0078704
6: -0.0084564, -0.0025468, -0.0085823, -0.0025450, -0.0059114, 0.0060356
7: -0.0156142, -0.0097188, -0.0156887, -0.0099366, -0.0056200, 0.0059107
8: -0.0147738, -0.0098101, -0.0147929, -0.0098611, -0.0049127, 0.0049828
9: -0.0020266, 0.0031579, -0.0021460, 0.0031706, -0.0051972, 0.0053040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_B2_A2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006472
time: 0.84 seconds

## Relational analysis of IS_B2_A2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006454
time: 0.90 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008449, 0.0064963, -0.0008555, 0.0064963, -0.0067004, 0.0067136
1: -0.0036091, 0.0000352, -0.0036091, 0.0000352, -0.0034813, 0.0034784
2: 0.0106146, 0.0166320, 0.0106145, 0.0166320, -0.0060174, 0.0060175
3: 1.0060627, 1.0071328, 1.0060413, 1.0071328, -0.0010700, 0.0010915
4: -0.0043504, -0.0024310, -0.0043508, -0.0024310, -0.0019194, 0.0019198
5: 0.0033365, 0.0117622, 0.0033290, 0.0117622, -0.0079203, 0.0079298
6: -0.0084564, -0.0025468, -0.0084564, -0.0025468, -0.0059096, 0.0059096
7: -0.0156142, -0.0097188, -0.0156142, -0.0096544, -0.0058902, 0.0058261
8: -0.0147738, -0.0098101, -0.0147852, -0.0097953, -0.0049784, 0.0049751
9: -0.0020266, 0.0031579, -0.0020268, 0.0031681, -0.0051947, 0.0051848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_B2_A2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006484
time: 0.80 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006484
time: 0.96 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.19 seconds
IS_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008203, upper bound: 0.0008151
IS_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
IS_B1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008024, upper bound: 0.0007982
IS_B1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008024, upper bound: 0.0007978
IS_B1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0007978
IS_B1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007978
IS_B1_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008197
IS_B1_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008197
IS_B1_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
IS_B1_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
IS_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
IS_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
IS_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
IS_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
IS_B2_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006652, upper bound: 0.0006472
IS_B2_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006652, upper bound: 0.0006472
IS_B2_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006652, upper bound: 0.0006484
IS_B2_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006652, upper bound: 0.0006484
IS_B2_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006472
IS_B2_A2_A2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006454
IS_B2_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006484
IS_B2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006484

## BFS IS instance: IS_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005172, 0.0087532, -0.0005172, 0.0087532, -0.0087020, 0.0087020
1: -0.0035757, 0.0013898, -0.0035757, 0.0013898, -0.0048279, 0.0048279
2: 0.0082893, 0.0167834, 0.0082893, 0.0167834, -0.0084941, 0.0084941
3: 1.0059873, 1.0071342, 1.0059873, 1.0071342, -0.0011469, 0.0011469
4: -0.0043712, -0.0016617, -0.0043712, -0.0016617, -0.0027094, 0.0027094
5: 0.0035854, 0.0146464, 0.0035854, 0.0146464, -0.0106129, 0.0106129
6: -0.0108744, -0.0025394, -0.0108744, -0.0025394, -0.0083350, 0.0083350
7: -0.0166883, -0.0104205, -0.0166883, -0.0104205, -0.0062045, 0.0062044
8: -0.0149728, -0.0085120, -0.0149728, -0.0085120, -0.0064608, 0.0064608
9: -0.0040546, 0.0032390, -0.0040546, 0.0032390, -0.0072936, 0.0072936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008203
time: 0.93 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005172, 0.0087532, -0.0006974, 0.0085953, -0.0085496, 0.0088861
1: -0.0035757, 0.0013898, -0.0036084, 0.0012982, -0.0047384, 0.0048463
2: 0.0082893, 0.0167834, 0.0084613, 0.0167772, -0.0084879, 0.0083221
3: 1.0059873, 1.0071342, 1.0059916, 1.0071584, -0.0011711, 0.0011426
4: -0.0043712, -0.0016617, -0.0043705, -0.0017180, -0.0026532, 0.0027088
5: 0.0035854, 0.0146464, 0.0034470, 0.0144438, -0.0104139, 0.0107544
6: -0.0108744, -0.0025394, -0.0107042, -0.0025408, -0.0083337, 0.0081648
7: -0.0166883, -0.0104205, -0.0165982, -0.0101471, -0.0064779, 0.0061153
8: -0.0149728, -0.0085120, -0.0149665, -0.0086243, -0.0063484, 0.0064546
9: -0.0040546, 0.0032390, -0.0038994, 0.0032361, -0.0072907, 0.0071384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008024
time: 1.06 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0008024
time: 0.88 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0005077, 0.0087532, -0.0088576, 0.0085387
1: -0.0036000, 0.0012982, -0.0035726, 0.0013898, -0.0048364, 0.0047345
2: 0.0084613, 0.0167772, 0.0082893, 0.0167834, -0.0083221, 0.0084879
3: 1.0059929, 1.0071430, 1.0059879, 1.0071290, -0.0011361, 0.0011551
4: -0.0043705, -0.0017180, -0.0043712, -0.0016617, -0.0027088, 0.0026532
5: 0.0034685, 0.0144438, 0.0035928, 0.0146464, -0.0107323, 0.0104054
6: -0.0107042, -0.0025416, -0.0108744, -0.0025397, -0.0081645, 0.0083328
7: -0.0165982, -0.0101711, -0.0166883, -0.0104289, -0.0061068, 0.0064536
8: -0.0149665, -0.0086314, -0.0149728, -0.0085126, -0.0064539, 0.0063414
9: -0.0038993, 0.0032361, -0.0040545, 0.0032390, -0.0071383, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007982
time: 1.02 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007982
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0005026, 0.0087532, -0.0088565, 0.0088841
1: -0.0035939, 0.0015108, -0.0035697, 0.0013898, -0.0048375, 0.0049463
2: 0.0081186, 0.0168105, 0.0082893, 0.0167834, -0.0086649, 0.0085212
3: 1.0059661, 1.0071237, 1.0059880, 1.0071189, -0.0011529, 0.0011357
4: -0.0043762, -0.0016037, -0.0043712, -0.0016617, -0.0027145, 0.0027675
5: 0.0034765, 0.0148873, 0.0035968, 0.0146464, -0.0107316, 0.0108475
6: -0.0110755, -0.0025432, -0.0108744, -0.0025405, -0.0085350, 0.0083313
7: -0.0167340, -0.0101752, -0.0166883, -0.0104316, -0.0062403, 0.0064506
8: -0.0150302, -0.0084228, -0.0149728, -0.0085129, -0.0065173, 0.0065499
9: -0.0041862, 0.0032678, -0.0040545, 0.0032390, -0.0074252, 0.0073224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007978
time: 0.97 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007978
time: 0.92 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005172, 0.0087532, -0.0006698, 0.0085953, -0.0085496, 0.0088576
1: -0.0035757, 0.0013898, -0.0036000, 0.0012982, -0.0047383, 0.0048363
2: 0.0082893, 0.0167834, 0.0084613, 0.0167772, -0.0084879, 0.0083221
3: 1.0059873, 1.0071342, 1.0059929, 1.0071430, -0.0011557, 0.0011413
4: -0.0043712, -0.0016617, -0.0043705, -0.0017180, -0.0026532, 0.0027088
5: 0.0035854, 0.0146464, 0.0034685, 0.0144438, -0.0104139, 0.0107323
6: -0.0108744, -0.0025394, -0.0107042, -0.0025416, -0.0083328, 0.0081648
7: -0.0166883, -0.0104205, -0.0165982, -0.0101711, -0.0064536, 0.0061152
8: -0.0149728, -0.0085120, -0.0149665, -0.0086314, -0.0063414, 0.0064546
9: -0.0040546, 0.0032390, -0.0038993, 0.0032361, -0.0072907, 0.0071383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 0.82 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006974, 0.0085953, -0.0006698, 0.0085953, -0.0086219, 0.0085905
1: -0.0036084, 0.0012982, -0.0036000, 0.0012982, -0.0047441, 0.0047324
2: 0.0084613, 0.0167772, 0.0084613, 0.0167772, -0.0083159, 0.0083159
3: 1.0059916, 1.0071584, 1.0059929, 1.0071430, -0.0011514, 0.0011655
4: -0.0043705, -0.0017180, -0.0043705, -0.0017180, -0.0026525, 0.0026525
5: 0.0034470, 0.0144438, 0.0034685, 0.0144438, -0.0104683, 0.0104438
6: -0.0107042, -0.0025408, -0.0107042, -0.0025416, -0.0081626, 0.0081634
7: -0.0165982, -0.0101471, -0.0165982, -0.0101711, -0.0063526, 0.0063771
8: -0.0149665, -0.0086243, -0.0149665, -0.0086314, -0.0063351, 0.0063422
9: -0.0038994, 0.0032361, -0.0038993, 0.0032361, -0.0071356, 0.0071355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.92 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005172, 0.0087532, -0.0006596, 0.0089437, -0.0088956, 0.0088565
1: -0.0035757, 0.0013898, -0.0035939, 0.0015108, -0.0049515, 0.0048375
2: 0.0082893, 0.0167834, 0.0081186, 0.0168105, -0.0085212, 0.0086649
3: 1.0059873, 1.0071342, 1.0059661, 1.0071237, -0.0011364, 0.0011681
4: -0.0043712, -0.0016617, -0.0043762, -0.0016037, -0.0027675, 0.0027145
5: 0.0035854, 0.0146464, 0.0034765, 0.0148873, -0.0108564, 0.0107315
6: -0.0108744, -0.0025394, -0.0110755, -0.0025432, -0.0083313, 0.0085361
7: -0.0166883, -0.0104205, -0.0167340, -0.0101752, -0.0064506, 0.0062512
8: -0.0149728, -0.0085120, -0.0150302, -0.0084228, -0.0065499, 0.0065182
9: -0.0040546, 0.0032390, -0.0041862, 0.0032678, -0.0073224, 0.0074252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 0.82 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006974, 0.0085953, -0.0006596, 0.0089437, -0.0089725, 0.0085887
1: -0.0036084, 0.0012982, -0.0035939, 0.0015108, -0.0049605, 0.0047321
2: 0.0084613, 0.0167772, 0.0081186, 0.0168105, -0.0083492, 0.0086587
3: 1.0059916, 1.0071584, 1.0059661, 1.0071237, -0.0011321, 0.0011923
4: -0.0043705, -0.0017180, -0.0043762, -0.0016037, -0.0027668, 0.0026582
5: 0.0034470, 0.0144438, 0.0034765, 0.0148873, -0.0109135, 0.0104425
6: -0.0107042, -0.0025408, -0.0110755, -0.0025432, -0.0081611, 0.0085347
7: -0.0165982, -0.0101471, -0.0167340, -0.0101752, -0.0063497, 0.0065138
8: -0.0149665, -0.0086243, -0.0150302, -0.0084228, -0.0065437, 0.0064058
9: -0.0038994, 0.0032361, -0.0041862, 0.0032678, -0.0071673, 0.0074223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
time: 1.06 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
time: 1.06 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0005077, 0.0087532, -0.0088576, 0.0085387
1: -0.0036000, 0.0012982, -0.0035726, 0.0013898, -0.0048364, 0.0047345
2: 0.0084613, 0.0167772, 0.0082893, 0.0167834, -0.0083221, 0.0084879
3: 1.0059929, 1.0071430, 1.0059879, 1.0071290, -0.0011361, 0.0011551
4: -0.0043705, -0.0017180, -0.0043712, -0.0016617, -0.0027088, 0.0026532
5: 0.0034685, 0.0144438, 0.0035928, 0.0146464, -0.0107323, 0.0104054
6: -0.0107042, -0.0025416, -0.0108744, -0.0025397, -0.0081645, 0.0083328
7: -0.0165982, -0.0101711, -0.0166883, -0.0104289, -0.0061068, 0.0064536
8: -0.0149665, -0.0086314, -0.0149728, -0.0085126, -0.0064539, 0.0063414
9: -0.0038993, 0.0032361, -0.0040545, 0.0032390, -0.0071383, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 0.99 seconds

## Relational analysis of IS_B1_A1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0005026, 0.0087532, -0.0088565, 0.0088841
1: -0.0035939, 0.0015108, -0.0035697, 0.0013898, -0.0048375, 0.0049463
2: 0.0081186, 0.0168105, 0.0082893, 0.0167834, -0.0086649, 0.0085212
3: 1.0059661, 1.0071237, 1.0059880, 1.0071189, -0.0011529, 0.0011357
4: -0.0043762, -0.0016037, -0.0043712, -0.0016617, -0.0027145, 0.0027675
5: 0.0034765, 0.0148873, 0.0035968, 0.0146464, -0.0107316, 0.0108475
6: -0.0110755, -0.0025432, -0.0108744, -0.0025405, -0.0085350, 0.0083313
7: -0.0167340, -0.0101752, -0.0166883, -0.0104316, -0.0062403, 0.0064506
8: -0.0150302, -0.0084228, -0.0149728, -0.0085129, -0.0065173, 0.0065499
9: -0.0041862, 0.0032678, -0.0040545, 0.0032390, -0.0074252, 0.0073224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.98 seconds

## Relational analysis of IS_B1_A1_A2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007974
time: 0.87 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006698, 0.0085953, -0.0085905, 0.0085905
1: -0.0036000, 0.0012982, -0.0036000, 0.0012982, -0.0047324, 0.0047324
2: 0.0084613, 0.0167772, 0.0084613, 0.0167772, -0.0083159, 0.0083159
3: 1.0059929, 1.0071430, 1.0059929, 1.0071430, -0.0011501, 0.0011501
4: -0.0043705, -0.0017180, -0.0043705, -0.0017180, -0.0026525, 0.0026525
5: 0.0034685, 0.0144438, 0.0034685, 0.0144438, -0.0104438, 0.0104438
6: -0.0107042, -0.0025416, -0.0107042, -0.0025416, -0.0081626, 0.0081626
7: -0.0165982, -0.0101711, -0.0165982, -0.0101711, -0.0063526, 0.0063526
8: -0.0149665, -0.0086314, -0.0149665, -0.0086314, -0.0063351, 0.0063351
9: -0.0038993, 0.0032361, -0.0038993, 0.0032361, -0.0071355, 0.0071355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 1.03 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0007982
time: 1.18 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006596, 0.0089437, -0.0089411, 0.0085923
1: -0.0036000, 0.0012982, -0.0035939, 0.0015108, -0.0049488, 0.0047332
2: 0.0084613, 0.0167772, 0.0081186, 0.0168105, -0.0083492, 0.0086587
3: 1.0059929, 1.0071430, 1.0059661, 1.0071237, -0.0011308, 0.0011770
4: -0.0043705, -0.0017180, -0.0043762, -0.0016037, -0.0027668, 0.0026582
5: 0.0034685, 0.0144438, 0.0034765, 0.0148873, -0.0108891, 0.0104457
6: -0.0107042, -0.0025416, -0.0110755, -0.0025432, -0.0081611, 0.0085339
7: -0.0165982, -0.0101711, -0.0167340, -0.0101752, -0.0063495, 0.0064893
8: -0.0149665, -0.0086314, -0.0150302, -0.0084228, -0.0065437, 0.0063988
9: -0.0038993, 0.0032361, -0.0041862, 0.0032678, -0.0071672, 0.0074223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_B1_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 1.13 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2_B2_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
time: 0.96 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0005077, 0.0087532, -0.0088576, 0.0085387
1: -0.0036000, 0.0012982, -0.0035726, 0.0013898, -0.0048364, 0.0047345
2: 0.0084613, 0.0167772, 0.0082893, 0.0167834, -0.0083221, 0.0084879
3: 1.0059929, 1.0071430, 1.0059879, 1.0071290, -0.0011361, 0.0011551
4: -0.0043705, -0.0017180, -0.0043712, -0.0016617, -0.0027088, 0.0026532
5: 0.0034685, 0.0144438, 0.0035928, 0.0146464, -0.0107323, 0.0104054
6: -0.0107042, -0.0025416, -0.0108744, -0.0025397, -0.0081645, 0.0083328
7: -0.0165982, -0.0101711, -0.0166883, -0.0104289, -0.0061068, 0.0064536
8: -0.0149665, -0.0086314, -0.0149728, -0.0085126, -0.0064539, 0.0063414
9: -0.0038993, 0.0032361, -0.0040545, 0.0032390, -0.0071383, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 0.91 seconds

## Relational analysis of IS_B1_A1_A2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 1.00 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0005026, 0.0087532, -0.0088565, 0.0088841
1: -0.0035939, 0.0015108, -0.0035697, 0.0013898, -0.0048375, 0.0049463
2: 0.0081186, 0.0168105, 0.0082893, 0.0167834, -0.0086649, 0.0085212
3: 1.0059661, 1.0071237, 1.0059880, 1.0071189, -0.0011529, 0.0011357
4: -0.0043762, -0.0016037, -0.0043712, -0.0016617, -0.0027145, 0.0027675
5: 0.0034765, 0.0148873, 0.0035968, 0.0146464, -0.0107316, 0.0108475
6: -0.0110755, -0.0025432, -0.0108744, -0.0025405, -0.0085350, 0.0083313
7: -0.0167340, -0.0101752, -0.0166883, -0.0104316, -0.0062403, 0.0064506
8: -0.0150302, -0.0084228, -0.0149728, -0.0085129, -0.0065173, 0.0065499
9: -0.0041862, 0.0032678, -0.0040545, 0.0032390, -0.0074252, 0.0073224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
time: 0.97 seconds

## Relational analysis of IS_B1_A1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006698, 0.0085953, -0.0085923, 0.0089411
1: -0.0035939, 0.0015108, -0.0036000, 0.0012982, -0.0047332, 0.0049488
2: 0.0081186, 0.0168105, 0.0084613, 0.0167772, -0.0086587, 0.0083492
3: 1.0059661, 1.0071237, 1.0059929, 1.0071430, -0.0011770, 0.0011308
4: -0.0043762, -0.0016037, -0.0043705, -0.0017180, -0.0026582, 0.0027668
5: 0.0034765, 0.0148873, 0.0034685, 0.0144438, -0.0104457, 0.0108891
6: -0.0110755, -0.0025432, -0.0107042, -0.0025416, -0.0085339, 0.0081611
7: -0.0167340, -0.0101752, -0.0165982, -0.0101711, -0.0064893, 0.0063495
8: -0.0150302, -0.0084228, -0.0149665, -0.0086314, -0.0063988, 0.0065437
9: -0.0041862, 0.0032678, -0.0038993, 0.0032361, -0.0074223, 0.0071672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
time: 0.91 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007986
time: 0.88 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006596, 0.0089437, -0.0089189, 0.0089189
1: -0.0035939, 0.0015108, -0.0035939, 0.0015108, -0.0049383, 0.0049383
2: 0.0081186, 0.0168105, 0.0081186, 0.0168105, -0.0086919, 0.0086919
3: 1.0059661, 1.0071237, 1.0059661, 1.0071237, -0.0011576, 0.0011576
4: -0.0043762, -0.0016037, -0.0043762, -0.0016037, -0.0027725, 0.0027725
5: 0.0034765, 0.0148873, 0.0034765, 0.0148873, -0.0108717, 0.0108717
6: -0.0110755, -0.0025432, -0.0110755, -0.0025432, -0.0085323, 0.0085323
7: -0.0167340, -0.0101752, -0.0167340, -0.0101752, -0.0064838, 0.0064838
8: -0.0150302, -0.0084228, -0.0150302, -0.0084228, -0.0066073, 0.0066073
9: -0.0041862, 0.0032678, -0.0041862, 0.0032678, -0.0074540, 0.0074540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A2_A2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
time: 1.09 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007986
time: 0.87 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0005080, 0.0087532, -0.0088649, 0.0082496
1: -0.0035670, 0.0011276, -0.0035728, 0.0013898, -0.0048295, 0.0045619
2: 0.0087861, 0.0167752, 0.0082893, 0.0167834, -0.0079973, 0.0084859
3: 1.0059211, 1.0070909, 1.0059881, 1.0071293, -0.0012082, 0.0011028
4: -0.0043727, -0.0018248, -0.0043712, -0.0016617, -0.0027110, 0.0025464
5: 0.0034913, 0.0140638, 0.0035925, 0.0146464, -0.0107347, 0.0100305
6: -0.0103848, -0.0025434, -0.0108744, -0.0025397, -0.0078451, 0.0083311
7: -0.0164270, -0.0099621, -0.0166883, -0.0104285, -0.0059379, 0.0066694
8: -0.0150110, -0.0088693, -0.0149728, -0.0085126, -0.0064985, 0.0061034
9: -0.0036054, 0.0032739, -0.0040545, 0.0032390, -0.0068444, 0.0073284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.83 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0005046, 0.0087532, -0.0088676, 0.0085686
1: -0.0035629, 0.0013207, -0.0035705, 0.0013898, -0.0048307, 0.0047573
2: 0.0084764, 0.0168061, 0.0082893, 0.0167834, -0.0083071, 0.0085168
3: 1.0058957, 1.0070760, 1.0059879, 1.0071208, -0.0012251, 0.0010881
4: -0.0043781, -0.0017212, -0.0043712, -0.0016617, -0.0027164, 0.0026500
5: 0.0034944, 0.0144650, 0.0035953, 0.0146464, -0.0107368, 0.0104353
6: -0.0107207, -0.0025446, -0.0108744, -0.0025404, -0.0081803, 0.0083298
7: -0.0165489, -0.0099628, -0.0166883, -0.0104301, -0.0060587, 0.0066686
8: -0.0150734, -0.0086872, -0.0149728, -0.0085127, -0.0065606, 0.0062856
9: -0.0038642, 0.0033052, -0.0040545, 0.0032390, -0.0071032, 0.0073597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.84 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.13 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0006888, 0.0085953, -0.0087219, 0.0084352
1: -0.0035670, 0.0011276, -0.0036057, 0.0012982, -0.0047433, 0.0045807
2: 0.0087861, 0.0167752, 0.0084613, 0.0167772, -0.0079911, 0.0083139
3: 1.0059211, 1.0070909, 1.0059921, 1.0071536, -0.0012325, 0.0010989
4: -0.0043727, -0.0018248, -0.0043705, -0.0017180, -0.0026547, 0.0025457
5: 0.0034913, 0.0140638, 0.0034537, 0.0144437, -0.0105444, 0.0101732
6: -0.0103848, -0.0025434, -0.0107042, -0.0025410, -0.0078437, 0.0081609
7: -0.0164270, -0.0099621, -0.0165982, -0.0101546, -0.0062118, 0.0065814
8: -0.0150110, -0.0088693, -0.0149665, -0.0086267, -0.0063844, 0.0060972
9: -0.0036054, 0.0032739, -0.0038994, 0.0032361, -0.0068415, 0.0071733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.91 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0006834, 0.0085953, -0.0087227, 0.0087492
1: -0.0035629, 0.0013207, -0.0036029, 0.0012982, -0.0047436, 0.0047749
2: 0.0084764, 0.0168061, 0.0084613, 0.0167772, -0.0083009, 0.0083448
3: 1.0058957, 1.0070760, 1.0059923, 1.0071450, -0.0012493, 0.0010837
4: -0.0043781, -0.0017212, -0.0043705, -0.0017180, -0.0026601, 0.0026493
5: 0.0034944, 0.0144650, 0.0034580, 0.0144438, -0.0105450, 0.0105741
6: -0.0107207, -0.0025446, -0.0107043, -0.0025417, -0.0081790, 0.0081596
7: -0.0165489, -0.0099628, -0.0165982, -0.0101576, -0.0063314, 0.0065803
8: -0.0150734, -0.0086872, -0.0149665, -0.0086297, -0.0064437, 0.0062793
9: -0.0038642, 0.0033052, -0.0038994, 0.0032361, -0.0071004, 0.0072046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.12 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.18 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0005080, 0.0087532, -0.0088649, 0.0082496
1: -0.0035670, 0.0011276, -0.0035728, 0.0013898, -0.0048295, 0.0045619
2: 0.0087861, 0.0167752, 0.0082893, 0.0167834, -0.0079973, 0.0084859
3: 1.0059211, 1.0070909, 1.0059881, 1.0071293, -0.0012082, 0.0011028
4: -0.0043727, -0.0018248, -0.0043712, -0.0016617, -0.0027110, 0.0025464
5: 0.0034913, 0.0140638, 0.0035925, 0.0146464, -0.0107347, 0.0100305
6: -0.0103848, -0.0025434, -0.0108744, -0.0025397, -0.0078451, 0.0083311
7: -0.0164270, -0.0099621, -0.0166883, -0.0104285, -0.0059379, 0.0066694
8: -0.0150110, -0.0088693, -0.0149728, -0.0085126, -0.0064985, 0.0061034
9: -0.0036054, 0.0032739, -0.0040545, 0.0032390, -0.0068444, 0.0073284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.81 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.90 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0005046, 0.0087532, -0.0088676, 0.0085686
1: -0.0035629, 0.0013207, -0.0035705, 0.0013898, -0.0048307, 0.0047573
2: 0.0084764, 0.0168061, 0.0082893, 0.0167834, -0.0083071, 0.0085168
3: 1.0058957, 1.0070760, 1.0059879, 1.0071208, -0.0012251, 0.0010881
4: -0.0043781, -0.0017212, -0.0043712, -0.0016617, -0.0027164, 0.0026500
5: 0.0034944, 0.0144650, 0.0035953, 0.0146464, -0.0107368, 0.0104353
6: -0.0107207, -0.0025446, -0.0108744, -0.0025404, -0.0081803, 0.0083298
7: -0.0165489, -0.0099628, -0.0166883, -0.0104301, -0.0060587, 0.0066686
8: -0.0150734, -0.0086872, -0.0149728, -0.0085127, -0.0065606, 0.0062856
9: -0.0038642, 0.0033052, -0.0040545, 0.0032390, -0.0071032, 0.0073597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.87 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.88 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0006888, 0.0085953, -0.0087219, 0.0084352
1: -0.0035670, 0.0011276, -0.0036057, 0.0012982, -0.0047433, 0.0045807
2: 0.0087861, 0.0167752, 0.0084613, 0.0167772, -0.0079911, 0.0083139
3: 1.0059211, 1.0070909, 1.0059921, 1.0071536, -0.0012325, 0.0010989
4: -0.0043727, -0.0018248, -0.0043705, -0.0017180, -0.0026547, 0.0025457
5: 0.0034913, 0.0140638, 0.0034537, 0.0144437, -0.0105444, 0.0101732
6: -0.0103848, -0.0025434, -0.0107042, -0.0025410, -0.0078437, 0.0081609
7: -0.0164270, -0.0099621, -0.0165982, -0.0101546, -0.0062118, 0.0065814
8: -0.0150110, -0.0088693, -0.0149665, -0.0086267, -0.0063844, 0.0060972
9: -0.0036054, 0.0032739, -0.0038994, 0.0032361, -0.0068415, 0.0071733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.91 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0006834, 0.0085953, -0.0087227, 0.0087492
1: -0.0035629, 0.0013207, -0.0036029, 0.0012982, -0.0047436, 0.0047749
2: 0.0084764, 0.0168061, 0.0084613, 0.0167772, -0.0083009, 0.0083448
3: 1.0058957, 1.0070760, 1.0059923, 1.0071450, -0.0012493, 0.0010837
4: -0.0043781, -0.0017212, -0.0043705, -0.0017180, -0.0026601, 0.0026493
5: 0.0034944, 0.0144650, 0.0034580, 0.0144438, -0.0105450, 0.0105741
6: -0.0107207, -0.0025446, -0.0107043, -0.0025417, -0.0081790, 0.0081596
7: -0.0165489, -0.0099628, -0.0165982, -0.0101576, -0.0063314, 0.0065803
8: -0.0150734, -0.0086872, -0.0149665, -0.0086297, -0.0064437, 0.0062793
9: -0.0038642, 0.0033052, -0.0038994, 0.0032361, -0.0071004, 0.0072046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.87 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.87 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0005080, 0.0087532, -0.0088649, 0.0082496
1: -0.0035670, 0.0011276, -0.0035728, 0.0013898, -0.0048295, 0.0045619
2: 0.0087861, 0.0167752, 0.0082893, 0.0167834, -0.0079973, 0.0084859
3: 1.0059211, 1.0070909, 1.0059881, 1.0071293, -0.0012082, 0.0011028
4: -0.0043727, -0.0018248, -0.0043712, -0.0016617, -0.0027110, 0.0025464
5: 0.0034913, 0.0140638, 0.0035925, 0.0146464, -0.0107347, 0.0100305
6: -0.0103848, -0.0025434, -0.0108744, -0.0025397, -0.0078451, 0.0083311
7: -0.0164270, -0.0099621, -0.0166883, -0.0104285, -0.0059379, 0.0066694
8: -0.0150110, -0.0088693, -0.0149728, -0.0085126, -0.0064985, 0.0061034
9: -0.0036054, 0.0032739, -0.0040545, 0.0032390, -0.0068444, 0.0073284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.82 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.90 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0005046, 0.0087532, -0.0088676, 0.0085686
1: -0.0035629, 0.0013207, -0.0035705, 0.0013898, -0.0048307, 0.0047573
2: 0.0084764, 0.0168061, 0.0082893, 0.0167834, -0.0083071, 0.0085168
3: 1.0058957, 1.0070760, 1.0059879, 1.0071208, -0.0012251, 0.0010881
4: -0.0043781, -0.0017212, -0.0043712, -0.0016617, -0.0027164, 0.0026500
5: 0.0034944, 0.0144650, 0.0035953, 0.0146464, -0.0107368, 0.0104353
6: -0.0107207, -0.0025446, -0.0108744, -0.0025404, -0.0081803, 0.0083298
7: -0.0165489, -0.0099628, -0.0166883, -0.0104301, -0.0060587, 0.0066686
8: -0.0150734, -0.0086872, -0.0149728, -0.0085127, -0.0065606, 0.0062856
9: -0.0038642, 0.0033052, -0.0040545, 0.0032390, -0.0071032, 0.0073597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.85 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.13 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0006888, 0.0085953, -0.0087219, 0.0084352
1: -0.0035670, 0.0011276, -0.0036057, 0.0012982, -0.0047433, 0.0045807
2: 0.0087861, 0.0167752, 0.0084613, 0.0167772, -0.0079911, 0.0083139
3: 1.0059211, 1.0070909, 1.0059921, 1.0071536, -0.0012325, 0.0010989
4: -0.0043727, -0.0018248, -0.0043705, -0.0017180, -0.0026547, 0.0025457
5: 0.0034913, 0.0140638, 0.0034537, 0.0144437, -0.0105444, 0.0101732
6: -0.0103848, -0.0025434, -0.0107042, -0.0025410, -0.0078437, 0.0081609
7: -0.0164270, -0.0099621, -0.0165982, -0.0101546, -0.0062118, 0.0065814
8: -0.0150110, -0.0088693, -0.0149665, -0.0086267, -0.0063844, 0.0060972
9: -0.0036054, 0.0032739, -0.0038994, 0.0032361, -0.0068415, 0.0071733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.91 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0006834, 0.0085953, -0.0087227, 0.0087492
1: -0.0035629, 0.0013207, -0.0036029, 0.0012982, -0.0047436, 0.0047749
2: 0.0084764, 0.0168061, 0.0084613, 0.0167772, -0.0083009, 0.0083448
3: 1.0058957, 1.0070760, 1.0059923, 1.0071450, -0.0012493, 0.0010837
4: -0.0043781, -0.0017212, -0.0043705, -0.0017180, -0.0026601, 0.0026493
5: 0.0034944, 0.0144650, 0.0034580, 0.0144438, -0.0105450, 0.0105741
6: -0.0107207, -0.0025446, -0.0107043, -0.0025417, -0.0081790, 0.0081596
7: -0.0165489, -0.0099628, -0.0165982, -0.0101576, -0.0063314, 0.0065803
8: -0.0150734, -0.0086872, -0.0149665, -0.0086297, -0.0064437, 0.0062793
9: -0.0038642, 0.0033052, -0.0038994, 0.0032361, -0.0071004, 0.0072046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.12 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 1.06 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0005080, 0.0087532, -0.0088649, 0.0082496
1: -0.0035670, 0.0011276, -0.0035728, 0.0013898, -0.0048295, 0.0045619
2: 0.0087861, 0.0167752, 0.0082893, 0.0167834, -0.0079973, 0.0084859
3: 1.0059211, 1.0070909, 1.0059881, 1.0071293, -0.0012082, 0.0011028
4: -0.0043727, -0.0018248, -0.0043712, -0.0016617, -0.0027110, 0.0025464
5: 0.0034913, 0.0140638, 0.0035925, 0.0146464, -0.0107347, 0.0100305
6: -0.0103848, -0.0025434, -0.0108744, -0.0025397, -0.0078451, 0.0083311
7: -0.0164270, -0.0099621, -0.0166883, -0.0104285, -0.0059379, 0.0066694
8: -0.0150110, -0.0088693, -0.0149728, -0.0085126, -0.0064985, 0.0061034
9: -0.0036054, 0.0032739, -0.0040545, 0.0032390, -0.0068444, 0.0073284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.80 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.90 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0005046, 0.0087532, -0.0088676, 0.0085686
1: -0.0035629, 0.0013207, -0.0035705, 0.0013898, -0.0048307, 0.0047573
2: 0.0084764, 0.0168061, 0.0082893, 0.0167834, -0.0083071, 0.0085168
3: 1.0058957, 1.0070760, 1.0059879, 1.0071208, -0.0012251, 0.0010881
4: -0.0043781, -0.0017212, -0.0043712, -0.0016617, -0.0027164, 0.0026500
5: 0.0034944, 0.0144650, 0.0035953, 0.0146464, -0.0107368, 0.0104353
6: -0.0107207, -0.0025446, -0.0108744, -0.0025404, -0.0081803, 0.0083298
7: -0.0165489, -0.0099628, -0.0166883, -0.0104301, -0.0060587, 0.0066686
8: -0.0150734, -0.0086872, -0.0149728, -0.0085127, -0.0065606, 0.0062856
9: -0.0038642, 0.0033052, -0.0040545, 0.0032390, -0.0071032, 0.0073597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.86 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.86 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006444, 0.0083002, -0.0006888, 0.0085953, -0.0087219, 0.0084352
1: -0.0035670, 0.0011276, -0.0036057, 0.0012982, -0.0047433, 0.0045807
2: 0.0087861, 0.0167752, 0.0084613, 0.0167772, -0.0079911, 0.0083139
3: 1.0059211, 1.0070909, 1.0059921, 1.0071536, -0.0012325, 0.0010989
4: -0.0043727, -0.0018248, -0.0043705, -0.0017180, -0.0026547, 0.0025457
5: 0.0034913, 0.0140638, 0.0034537, 0.0144437, -0.0105444, 0.0101732
6: -0.0103848, -0.0025434, -0.0107042, -0.0025410, -0.0078437, 0.0081609
7: -0.0164270, -0.0099621, -0.0165982, -0.0101546, -0.0062118, 0.0065814
8: -0.0150110, -0.0088693, -0.0149665, -0.0086267, -0.0063844, 0.0060972
9: -0.0036054, 0.0032739, -0.0038994, 0.0032361, -0.0068415, 0.0071733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.94 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006403, 0.0086152, -0.0006834, 0.0085953, -0.0087227, 0.0087492
1: -0.0035629, 0.0013207, -0.0036029, 0.0012982, -0.0047436, 0.0047749
2: 0.0084764, 0.0168061, 0.0084613, 0.0167772, -0.0083009, 0.0083448
3: 1.0058957, 1.0070760, 1.0059923, 1.0071450, -0.0012493, 0.0010837
4: -0.0043781, -0.0017212, -0.0043705, -0.0017180, -0.0026601, 0.0026493
5: 0.0034944, 0.0144650, 0.0034580, 0.0144438, -0.0105450, 0.0105741
6: -0.0107207, -0.0025446, -0.0107043, -0.0025417, -0.0081790, 0.0081596
7: -0.0165489, -0.0099628, -0.0165982, -0.0101576, -0.0063314, 0.0065803
8: -0.0150734, -0.0086872, -0.0149665, -0.0086297, -0.0064437, 0.0062793
9: -0.0038642, 0.0033052, -0.0038994, 0.0032361, -0.0071004, 0.0072046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.91 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 1.06 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005079, 0.0087532, -0.0006444, 0.0083002, -0.0082624, 0.0088649
1: -0.0035727, 0.0013898, -0.0035670, 0.0011276, -0.0045629, 0.0048295
2: 0.0082893, 0.0167834, 0.0087861, 0.0167752, -0.0084859, 0.0079973
3: 1.0059880, 1.0071293, 1.0059211, 1.0070909, -0.0011029, 0.0012082
4: -0.0043712, -0.0016617, -0.0043727, -0.0018248, -0.0025464, 0.0027110
5: 0.0035926, 0.0146464, 0.0034913, 0.0140638, -0.0100405, 0.0107347
6: -0.0108744, -0.0025397, -0.0103848, -0.0025434, -0.0083311, 0.0078451
7: -0.0166883, -0.0104289, -0.0164270, -0.0099621, -0.0066694, 0.0059391
8: -0.0149728, -0.0085126, -0.0150110, -0.0088693, -0.0061034, 0.0064984
9: -0.0040545, 0.0032390, -0.0036054, 0.0032739, -0.0073284, 0.0068444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.79 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005044, 0.0087532, -0.0006403, 0.0086152, -0.0085822, 0.0088676
1: -0.0035704, 0.0013898, -0.0035629, 0.0013207, -0.0047589, 0.0048307
2: 0.0082893, 0.0167834, 0.0084764, 0.0168061, -0.0085168, 0.0083071
3: 1.0059880, 1.0071208, 1.0058957, 1.0070760, -0.0010880, 0.0012251
4: -0.0043712, -0.0016617, -0.0043781, -0.0017212, -0.0026500, 0.0027164
5: 0.0035954, 0.0146464, 0.0034944, 0.0144650, -0.0104461, 0.0107368
6: -0.0108744, -0.0025404, -0.0107207, -0.0025446, -0.0083298, 0.0081803
7: -0.0166883, -0.0104305, -0.0165489, -0.0099628, -0.0066686, 0.0060600
8: -0.0149728, -0.0085128, -0.0150734, -0.0086872, -0.0062856, 0.0065606
9: -0.0040545, 0.0032390, -0.0038642, 0.0033052, -0.0073597, 0.0071032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 1.10 seconds

## Relational analysis of IS_B2_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 1.01 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006886, 0.0085954, -0.0006444, 0.0083002, -0.0084520, 0.0087219
1: -0.0036057, 0.0012982, -0.0035670, 0.0011276, -0.0045846, 0.0047433
2: 0.0084613, 0.0167772, 0.0087861, 0.0167752, -0.0083139, 0.0079911
3: 1.0059922, 1.0071536, 1.0059211, 1.0070909, -0.0010988, 0.0012325
4: -0.0043705, -0.0017180, -0.0043727, -0.0018248, -0.0025457, 0.0026547
5: 0.0034539, 0.0144438, 0.0034913, 0.0140638, -0.0101864, 0.0105444
6: -0.0107042, -0.0025410, -0.0103848, -0.0025434, -0.0081609, 0.0078437
7: -0.0165982, -0.0101550, -0.0164270, -0.0099621, -0.0065814, 0.0062132
8: -0.0149665, -0.0086267, -0.0150110, -0.0088693, -0.0060972, 0.0063843
9: -0.0038994, 0.0032361, -0.0036054, 0.0032739, -0.0071733, 0.0068415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.76 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006832, 0.0085953, -0.0006403, 0.0086152, -0.0087667, 0.0087227
1: -0.0036029, 0.0012982, -0.0035629, 0.0013207, -0.0047794, 0.0047436
2: 0.0084613, 0.0167772, 0.0084764, 0.0168061, -0.0083448, 0.0083009
3: 1.0059924, 1.0071450, 1.0058957, 1.0070760, -0.0010836, 0.0012493
4: -0.0043705, -0.0017180, -0.0043781, -0.0017212, -0.0026493, 0.0026601
5: 0.0034581, 0.0144438, 0.0034944, 0.0144650, -0.0105878, 0.0105450
6: -0.0107042, -0.0025417, -0.0107207, -0.0025446, -0.0081596, 0.0081790
7: -0.0165982, -0.0101579, -0.0165489, -0.0099628, -0.0065803, 0.0063328
8: -0.0149665, -0.0086297, -0.0150734, -0.0086872, -0.0062793, 0.0064437
9: -0.0038994, 0.0032361, -0.0038642, 0.0033052, -0.0072046, 0.0071004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 1.02 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 0.84 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005079, 0.0087532, -0.0006444, 0.0083002, -0.0082624, 0.0088649
1: -0.0035727, 0.0013898, -0.0035670, 0.0011276, -0.0045629, 0.0048295
2: 0.0082893, 0.0167834, 0.0087861, 0.0167752, -0.0084859, 0.0079973
3: 1.0059880, 1.0071293, 1.0059211, 1.0070909, -0.0011029, 0.0012082
4: -0.0043712, -0.0016617, -0.0043727, -0.0018248, -0.0025464, 0.0027110
5: 0.0035926, 0.0146464, 0.0034913, 0.0140638, -0.0100405, 0.0107347
6: -0.0108744, -0.0025397, -0.0103848, -0.0025434, -0.0083311, 0.0078451
7: -0.0166883, -0.0104289, -0.0164270, -0.0099621, -0.0066694, 0.0059391
8: -0.0149728, -0.0085126, -0.0150110, -0.0088693, -0.0061034, 0.0064984
9: -0.0040545, 0.0032390, -0.0036054, 0.0032739, -0.0073284, 0.0068444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
time: 0.75 seconds

## Relational analysis of IS_B2_A1_A1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.82 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005044, 0.0087532, -0.0006403, 0.0086152, -0.0085822, 0.0088676
1: -0.0035704, 0.0013898, -0.0035629, 0.0013207, -0.0047589, 0.0048307
2: 0.0082893, 0.0167834, 0.0084764, 0.0168061, -0.0085168, 0.0083071
3: 1.0059880, 1.0071208, 1.0058957, 1.0070760, -0.0010880, 0.0012251
4: -0.0043712, -0.0016617, -0.0043781, -0.0017212, -0.0026500, 0.0027164
5: 0.0035954, 0.0146464, 0.0034944, 0.0144650, -0.0104461, 0.0107368
6: -0.0108744, -0.0025404, -0.0107207, -0.0025446, -0.0083298, 0.0081803
7: -0.0166883, -0.0104305, -0.0165489, -0.0099628, -0.0066686, 0.0060600
8: -0.0149728, -0.0085128, -0.0150734, -0.0086872, -0.0062856, 0.0065606
9: -0.0040545, 0.0032390, -0.0038642, 0.0033052, -0.0073597, 0.0071032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 0.93 seconds

## Relational analysis of IS_B2_A1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 1.11 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006886, 0.0085954, -0.0006444, 0.0083002, -0.0084520, 0.0087219
1: -0.0036057, 0.0012982, -0.0035670, 0.0011276, -0.0045846, 0.0047433
2: 0.0084613, 0.0167772, 0.0087861, 0.0167752, -0.0083139, 0.0079911
3: 1.0059922, 1.0071536, 1.0059211, 1.0070909, -0.0010988, 0.0012325
4: -0.0043705, -0.0017180, -0.0043727, -0.0018248, -0.0025457, 0.0026547
5: 0.0034539, 0.0144438, 0.0034913, 0.0140638, -0.0101864, 0.0105444
6: -0.0107042, -0.0025410, -0.0103848, -0.0025434, -0.0081609, 0.0078437
7: -0.0165982, -0.0101550, -0.0164270, -0.0099621, -0.0065814, 0.0062132
8: -0.0149665, -0.0086267, -0.0150110, -0.0088693, -0.0060972, 0.0063843
9: -0.0038994, 0.0032361, -0.0036054, 0.0032739, -0.0071733, 0.0068415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.78 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006832, 0.0085953, -0.0006403, 0.0086152, -0.0087667, 0.0087227
1: -0.0036029, 0.0012982, -0.0035629, 0.0013207, -0.0047794, 0.0047436
2: 0.0084613, 0.0167772, 0.0084764, 0.0168061, -0.0083448, 0.0083009
3: 1.0059924, 1.0071450, 1.0058957, 1.0070760, -0.0010836, 0.0012493
4: -0.0043705, -0.0017180, -0.0043781, -0.0017212, -0.0026493, 0.0026601
5: 0.0034581, 0.0144438, 0.0034944, 0.0144650, -0.0105878, 0.0105450
6: -0.0107042, -0.0025417, -0.0107207, -0.0025446, -0.0081596, 0.0081790
7: -0.0165982, -0.0101579, -0.0165489, -0.0099628, -0.0065803, 0.0063328
8: -0.0149665, -0.0086297, -0.0150734, -0.0086872, -0.0062793, 0.0064437
9: -0.0038994, 0.0032361, -0.0038642, 0.0033052, -0.0072046, 0.0071004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 1.08 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 0.83 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005079, 0.0087532, -0.0006444, 0.0083002, -0.0082624, 0.0088649
1: -0.0035727, 0.0013898, -0.0035670, 0.0011276, -0.0045629, 0.0048295
2: 0.0082893, 0.0167834, 0.0087861, 0.0167752, -0.0084859, 0.0079973
3: 1.0059880, 1.0071293, 1.0059211, 1.0070909, -0.0011029, 0.0012082
4: -0.0043712, -0.0016617, -0.0043727, -0.0018248, -0.0025464, 0.0027110
5: 0.0035926, 0.0146464, 0.0034913, 0.0140638, -0.0100405, 0.0107347
6: -0.0108744, -0.0025397, -0.0103848, -0.0025434, -0.0083311, 0.0078451
7: -0.0166883, -0.0104289, -0.0164270, -0.0099621, -0.0066694, 0.0059391
8: -0.0149728, -0.0085126, -0.0150110, -0.0088693, -0.0061034, 0.0064984
9: -0.0040545, 0.0032390, -0.0036054, 0.0032739, -0.0073284, 0.0068444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.79 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005044, 0.0087532, -0.0006403, 0.0086152, -0.0085822, 0.0088676
1: -0.0035704, 0.0013898, -0.0035629, 0.0013207, -0.0047589, 0.0048307
2: 0.0082893, 0.0167834, 0.0084764, 0.0168061, -0.0085168, 0.0083071
3: 1.0059880, 1.0071208, 1.0058957, 1.0070760, -0.0010880, 0.0012251
4: -0.0043712, -0.0016617, -0.0043781, -0.0017212, -0.0026500, 0.0027164
5: 0.0035954, 0.0146464, 0.0034944, 0.0144650, -0.0104461, 0.0107368
6: -0.0108744, -0.0025404, -0.0107207, -0.0025446, -0.0083298, 0.0081803
7: -0.0166883, -0.0104305, -0.0165489, -0.0099628, -0.0066686, 0.0060600
8: -0.0149728, -0.0085128, -0.0150734, -0.0086872, -0.0062856, 0.0065606
9: -0.0040545, 0.0032390, -0.0038642, 0.0033052, -0.0073597, 0.0071032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 1.11 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 1.00 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006886, 0.0085954, -0.0006444, 0.0083002, -0.0084520, 0.0087219
1: -0.0036057, 0.0012982, -0.0035670, 0.0011276, -0.0045846, 0.0047433
2: 0.0084613, 0.0167772, 0.0087861, 0.0167752, -0.0083139, 0.0079911
3: 1.0059922, 1.0071536, 1.0059211, 1.0070909, -0.0010988, 0.0012325
4: -0.0043705, -0.0017180, -0.0043727, -0.0018248, -0.0025457, 0.0026547
5: 0.0034539, 0.0144438, 0.0034913, 0.0140638, -0.0101864, 0.0105444
6: -0.0107042, -0.0025410, -0.0103848, -0.0025434, -0.0081609, 0.0078437
7: -0.0165982, -0.0101550, -0.0164270, -0.0099621, -0.0065814, 0.0062132
8: -0.0149665, -0.0086267, -0.0150110, -0.0088693, -0.0060972, 0.0063843
9: -0.0038994, 0.0032361, -0.0036054, 0.0032739, -0.0071733, 0.0068415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.80 seconds

## Relational analysis of IS_B2_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
time: 0.76 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006832, 0.0085953, -0.0006403, 0.0086152, -0.0087667, 0.0087227
1: -0.0036029, 0.0012982, -0.0035629, 0.0013207, -0.0047794, 0.0047436
2: 0.0084613, 0.0167772, 0.0084764, 0.0168061, -0.0083448, 0.0083009
3: 1.0059924, 1.0071450, 1.0058957, 1.0070760, -0.0010836, 0.0012493
4: -0.0043705, -0.0017180, -0.0043781, -0.0017212, -0.0026493, 0.0026601
5: 0.0034581, 0.0144438, 0.0034944, 0.0144650, -0.0105878, 0.0105450
6: -0.0107042, -0.0025417, -0.0107207, -0.0025446, -0.0081596, 0.0081790
7: -0.0165982, -0.0101579, -0.0165489, -0.0099628, -0.0065803, 0.0063328
8: -0.0149665, -0.0086297, -0.0150734, -0.0086872, -0.0062793, 0.0064437
9: -0.0038994, 0.0032361, -0.0038642, 0.0033052, -0.0072046, 0.0071004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 1.03 seconds

## Relational analysis of IS_B2_A1_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
time: 0.88 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005079, 0.0087532, -0.0006444, 0.0083002, -0.0082624, 0.0088649
1: -0.0035727, 0.0013898, -0.0035670, 0.0011276, -0.0045629, 0.0048295
2: 0.0082893, 0.0167834, 0.0087861, 0.0167752, -0.0084859, 0.0079973
3: 1.0059880, 1.0071293, 1.0059211, 1.0070909, -0.0011029, 0.0012082
4: -0.0043712, -0.0016617, -0.0043727, -0.0018248, -0.0025464, 0.0027110
5: 0.0035926, 0.0146464, 0.0034913, 0.0140638, -0.0100405, 0.0107347
6: -0.0108744, -0.0025397, -0.0103848, -0.0025434, -0.0083311, 0.0078451
7: -0.0166883, -0.0104289, -0.0164270, -0.0099621, -0.0066694, 0.0059391
8: -0.0149728, -0.0085126, -0.0150110, -0.0088693, -0.0061034, 0.0064984
9: -0.0040545, 0.0032390, -0.0036054, 0.0032739, -0.0073284, 0.0068444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
time: 0.75 seconds

## Relational analysis of IS_B2_A1_A2_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.81 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005044, 0.0087532, -0.0006403, 0.0086152, -0.0085822, 0.0088676
1: -0.0035704, 0.0013898, -0.0035629, 0.0013207, -0.0047589, 0.0048307
2: 0.0082893, 0.0167834, 0.0084764, 0.0168061, -0.0085168, 0.0083071
3: 1.0059880, 1.0071208, 1.0058957, 1.0070760, -0.0010880, 0.0012251
4: -0.0043712, -0.0016617, -0.0043781, -0.0017212, -0.0026500, 0.0027164
5: 0.0035954, 0.0146464, 0.0034944, 0.0144650, -0.0104461, 0.0107368
6: -0.0108744, -0.0025404, -0.0107207, -0.0025446, -0.0083298, 0.0081803
7: -0.0166883, -0.0104305, -0.0165489, -0.0099628, -0.0066686, 0.0060600
8: -0.0149728, -0.0085128, -0.0150734, -0.0086872, -0.0062856, 0.0065606
9: -0.0040545, 0.0032390, -0.0038642, 0.0033052, -0.0073597, 0.0071032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
time: 0.93 seconds

## Relational analysis of IS_B2_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 1.07 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006886, 0.0085954, -0.0006444, 0.0083002, -0.0084520, 0.0087219
1: -0.0036057, 0.0012982, -0.0035670, 0.0011276, -0.0045846, 0.0047433
2: 0.0084613, 0.0167772, 0.0087861, 0.0167752, -0.0083139, 0.0079911
3: 1.0059922, 1.0071536, 1.0059211, 1.0070909, -0.0010988, 0.0012325
4: -0.0043705, -0.0017180, -0.0043727, -0.0018248, -0.0025457, 0.0026547
5: 0.0034539, 0.0144438, 0.0034913, 0.0140638, -0.0101864, 0.0105444
6: -0.0107042, -0.0025410, -0.0103848, -0.0025434, -0.0081609, 0.0078437
7: -0.0165982, -0.0101550, -0.0164270, -0.0099621, -0.0065814, 0.0062132
8: -0.0149665, -0.0086267, -0.0150110, -0.0088693, -0.0060972, 0.0063843
9: -0.0038994, 0.0032361, -0.0036054, 0.0032739, -0.0071733, 0.0068415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006832, 0.0085953, -0.0006403, 0.0086152, -0.0087667, 0.0087227
1: -0.0036029, 0.0012982, -0.0035629, 0.0013207, -0.0047794, 0.0047436
2: 0.0084613, 0.0167772, 0.0084764, 0.0168061, -0.0083448, 0.0083009
3: 1.0059924, 1.0071450, 1.0058957, 1.0070760, -0.0010836, 0.0012493
4: -0.0043705, -0.0017180, -0.0043781, -0.0017212, -0.0026493, 0.0026601
5: 0.0034581, 0.0144438, 0.0034944, 0.0144650, -0.0105878, 0.0105450
6: -0.0107042, -0.0025417, -0.0107207, -0.0025446, -0.0081596, 0.0081790
7: -0.0165982, -0.0101579, -0.0165489, -0.0099628, -0.0065803, 0.0063328
8: -0.0149665, -0.0086297, -0.0150734, -0.0086872, -0.0062793, 0.0064437
9: -0.0038994, 0.0032361, -0.0038642, 0.0033052, -0.0072046, 0.0071004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 64

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
time: 1.02 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
time: 0.87 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0006631, 0.0066133, -0.0006631, 0.0066133, -0.0067279, 0.0067279
1: -0.0035763, 0.0001045, -0.0035763, 0.0001045, -0.0035386, 0.0035386
2: 0.0104847, 0.0166393, 0.0104847, 0.0166393, -0.0061546, 0.0061546
3: 1.0060581, 1.0071082, 1.0060581, 1.0071082, -0.0010501, 0.0010501
4: -0.0043514, -0.0023889, -0.0043514, -0.0023889, -0.0019625, 0.0019625
5: 0.0034759, 0.0119122, 0.0034759, 0.0119122, -0.0080033, 0.0080033
6: -0.0085823, -0.0025450, -0.0085823, -0.0025450, -0.0060374, 0.0060374
7: -0.0156886, -0.0099997, -0.0156886, -0.0099997, -0.0056306, 0.0056306
8: -0.0147810, -0.0098703, -0.0147810, -0.0098703, -0.0049108, 0.0049108
9: -0.0021458, 0.0031600, -0.0021458, 0.0031600, -0.0053058, 0.0053058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A1_A1_B1_B1_A1

### Relational analysis result of IS_B2_A2_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006555, upper bound: 0.0006475
time: 0.85 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_B1_A2

### Relational analysis result of IS_B2_A2_A1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006555, upper bound: 0.0006475
time: 0.88 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0006631, 0.0066133, -0.0006535, 0.0070375, -0.0071541, 0.0067206
1: -0.0035763, 0.0001045, -0.0035844, 0.0003516, -0.0037918, 0.0035409
2: 0.0104847, 0.0166393, 0.0100238, 0.0166491, -0.0061644, 0.0066155
3: 1.0060581, 1.0071082, 1.0060570, 1.0071499, -0.0010918, 0.0010512
4: -0.0043514, -0.0023889, -0.0043507, -0.0022378, -0.0021136, 0.0019618
5: 0.0034759, 0.0119122, 0.0034826, 0.0124573, -0.0085494, 0.0079981
6: -0.0085823, -0.0025450, -0.0090402, -0.0025381, -0.0060442, 0.0064952
7: -0.0156886, -0.0099997, -0.0159388, -0.0100561, -0.0055756, 0.0058817
8: -0.0147810, -0.0098703, -0.0147593, -0.0095506, -0.0052304, 0.0048890
9: -0.0021458, 0.0031600, -0.0025707, 0.0031416, -0.0052875, 0.0057307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A1_A1_B1_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006555, upper bound: 0.0006475
time: 1.04 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006454
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0006631, 0.0066133, -0.0008449, 0.0064963, -0.0066147, 0.0069047
1: -0.0035763, 0.0001045, -0.0036091, 0.0000352, -0.0034720, 0.0035590
2: 0.0104847, 0.0166393, 0.0106146, 0.0166320, -0.0061473, 0.0060247
3: 1.0060581, 1.0071082, 1.0060627, 1.0071328, -0.0010747, 0.0010455
4: -0.0043514, -0.0023889, -0.0043504, -0.0024310, -0.0019204, 0.0019616
5: 0.0034759, 0.0119122, 0.0033365, 0.0117622, -0.0078559, 0.0081393
6: -0.0085823, -0.0025450, -0.0084564, -0.0025468, -0.0060356, 0.0059114
7: -0.0156886, -0.0099997, -0.0156142, -0.0097188, -0.0059099, 0.0055557
8: -0.0147810, -0.0098703, -0.0147738, -0.0098101, -0.0049710, 0.0049035
9: -0.0021458, 0.0031600, -0.0020266, 0.0031579, -0.0053038, 0.0051866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A1_A1_B2_B1_A1

### Relational analysis result of IS_B2_A2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006463
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_B1_A2

### Relational analysis result of IS_B2_A2_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006454
time: 0.77 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0006631, 0.0066133, -0.0008354, 0.0069174, -0.0070384, 0.0068960
1: -0.0035763, 0.0001045, -0.0036171, 0.0002796, -0.0037230, 0.0035636
2: 0.0104847, 0.0166393, 0.0101552, 0.0166417, -0.0061570, 0.0064840
3: 1.0060581, 1.0071082, 1.0060652, 1.0071748, -0.0011168, 0.0010430
4: -0.0043514, -0.0023889, -0.0043497, -0.0022806, -0.0020707, 0.0019609
5: 0.0034759, 0.0119122, 0.0033433, 0.0123033, -0.0083995, 0.0081332
6: -0.0085823, -0.0025450, -0.0089110, -0.0025404, -0.0060420, 0.0063661
7: -0.0156886, -0.0099997, -0.0158592, -0.0097729, -0.0058559, 0.0058009
8: -0.0147810, -0.0098703, -0.0147508, -0.0095482, -0.0052328, 0.0048806
9: -0.0021458, 0.0031600, -0.0024449, 0.0031378, -0.0052836, 0.0056049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A1_A1_B2_B2_A1

### Relational analysis result of IS_B2_A2_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006463
time: 0.80 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_B2_A2

### Relational analysis result of IS_B2_A2_A1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006454
time: 0.85 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008449, 0.0064963, -0.0006712, 0.0066133, -0.0069094, 0.0066297
1: -0.0036091, 0.0000352, -0.0035763, 0.0001045, -0.0035613, 0.0034720
2: 0.0106146, 0.0166320, 0.0104846, 0.0166393, -0.0060247, 0.0061474
3: 1.0060627, 1.0071328, 1.0060407, 1.0071082, -0.0010455, 0.0010921
4: -0.0043504, -0.0024310, -0.0043516, -0.0023889, -0.0019616, 0.0019207
5: 0.0033365, 0.0117622, 0.0034702, 0.0119122, -0.0081430, 0.0078670
6: -0.0084564, -0.0025468, -0.0085824, -0.0025450, -0.0059114, 0.0060356
7: -0.0156142, -0.0097188, -0.0156887, -0.0099514, -0.0056051, 0.0059105
8: -0.0147738, -0.0098101, -0.0147902, -0.0098632, -0.0049106, 0.0049801
9: -0.0020266, 0.0031579, -0.0021460, 0.0031682, -0.0051948, 0.0053039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006724, upper bound: 0.0006491
time: 1.06 seconds

## Relational analysis of IS_B2_A2_A2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006472
time: 0.93 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0008449, 0.0064963, -0.0008449, 0.0064963, -0.0066944, 0.0066944
1: -0.0036091, 0.0000352, -0.0036091, 0.0000352, -0.0034784, 0.0034784
2: 0.0106146, 0.0166320, 0.0106146, 0.0166320, -0.0060174, 0.0060174
3: 1.0060627, 1.0071328, 1.0060627, 1.0071328, -0.0010700, 0.0010700
4: -0.0043504, -0.0024310, -0.0043504, -0.0024310, -0.0019194, 0.0019194
5: 0.0033365, 0.0117622, 0.0033365, 0.0117622, -0.0079155, 0.0079155
6: -0.0084564, -0.0025468, -0.0084564, -0.0025468, -0.0059096, 0.0059096
7: -0.0156142, -0.0097188, -0.0156142, -0.0097188, -0.0058254, 0.0058254
8: -0.0147738, -0.0098101, -0.0147738, -0.0098101, -0.0049637, 0.0049637
9: -0.0020266, 0.0031579, -0.0020266, 0.0031579, -0.0051846, 0.0051846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A2_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006454
time: 1.02 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006463
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0008449, 0.0064963, -0.0008354, 0.0069174, -0.0071170, 0.0066867
1: -0.0036091, 0.0000352, -0.0036171, 0.0002796, -0.0037295, 0.0034811
2: 0.0106146, 0.0166320, 0.0101552, 0.0166417, -0.0060271, 0.0064768
3: 1.0060627, 1.0071328, 1.0060652, 1.0071748, -0.0011121, 0.0010675
4: -0.0043504, -0.0024310, -0.0043497, -0.0022806, -0.0020698, 0.0019187
5: 0.0033365, 0.0117622, 0.0033433, 0.0123033, -0.0084582, 0.0079101
6: -0.0084564, -0.0025468, -0.0089110, -0.0025404, -0.0059160, 0.0063643
7: -0.0156142, -0.0097188, -0.0158592, -0.0097729, -0.0057714, 0.0060705
8: -0.0147738, -0.0098101, -0.0147508, -0.0095482, -0.0052255, 0.0049407
9: -0.0020266, 0.0031579, -0.0024449, 0.0031378, -0.0051644, 0.0056028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_A2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A2_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006454
time: 1.07 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006463
time: 0.94 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.64 seconds
IS_B1_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008203
IS_B1_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
IS_B1_A1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008024
IS_B1_A1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0008024
IS_B1_A1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007982
IS_B1_A1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007982
IS_B1_A1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007978
IS_B1_A1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007978
IS_B1_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
IS_B1_A1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007974
IS_B1_A1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0007982
IS_B1_A1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
IS_B1_A1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
IS_B1_A1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
IS_B1_A1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007986
IS_B1_A1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007986
IS_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
IS_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
IS_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B2_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
IS_B2_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
IS_B2_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
IS_B2_A2_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006555, upper bound: 0.0006475
IS_B2_A2_A1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006555, upper bound: 0.0006475
IS_B2_A2_A1_A1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006555, upper bound: 0.0006475
IS_B2_A2_A1_A1_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006454
IS_B2_A2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006463
IS_B2_A2_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006454
IS_B2_A2_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006463
IS_B2_A2_A1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006454, upper bound: 0.0006454
IS_B2_A2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006724, upper bound: 0.0006491
IS_B2_A2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006472
IS_B2_A2_A2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006454
IS_B2_A2_A2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006463
IS_B2_A2_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006454
IS_B2_A2_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.64
Output dim: 3, lower bound: -0.0006463, upper bound: 0.0006463

## BFS IS instance: IS_B1_A1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005172, 0.0087532, -0.0005172, 0.0087532, -0.0087020, 0.0087020
1: -0.0035757, 0.0013898, -0.0035757, 0.0013898, -0.0048279, 0.0048279
2: 0.0082893, 0.0167834, 0.0082893, 0.0167834, -0.0084941, 0.0084941
3: 1.0059873, 1.0071342, 1.0059873, 1.0071342, -0.0011469, 0.0011469
4: -0.0043712, -0.0016617, -0.0043712, -0.0016617, -0.0027094, 0.0027094
5: 0.0035854, 0.0146464, 0.0035854, 0.0146464, -0.0106129, 0.0106129
6: -0.0108744, -0.0025394, -0.0108744, -0.0025394, -0.0083350, 0.0083350
7: -0.0166883, -0.0104205, -0.0166883, -0.0104205, -0.0062045, 0.0062044
8: -0.0149728, -0.0085120, -0.0149728, -0.0085120, -0.0064608, 0.0064608
9: -0.0040546, 0.0032390, -0.0040546, 0.0032390, -0.0072936, 0.0072936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008203, upper bound: 0.0008151
time: 0.96 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
time: 0.90 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006974, 0.0085953, -0.0005172, 0.0087532, -0.0088861, 0.0085496
1: -0.0036084, 0.0012982, -0.0035757, 0.0013898, -0.0048463, 0.0047384
2: 0.0084613, 0.0167772, 0.0082893, 0.0167834, -0.0083221, 0.0084879
3: 1.0059916, 1.0071584, 1.0059873, 1.0071342, -0.0011426, 0.0011711
4: -0.0043705, -0.0017180, -0.0043712, -0.0016617, -0.0027088, 0.0026532
5: 0.0034470, 0.0144438, 0.0035854, 0.0146464, -0.0107544, 0.0104139
6: -0.0107042, -0.0025408, -0.0108744, -0.0025394, -0.0081648, 0.0083337
7: -0.0165982, -0.0101471, -0.0166883, -0.0104205, -0.0061153, 0.0064779
8: -0.0149665, -0.0086243, -0.0149728, -0.0085120, -0.0064546, 0.0063484
9: -0.0038994, 0.0032361, -0.0040546, 0.0032390, -0.0071384, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008024, upper bound: 0.0007982
time: 0.83 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008024, upper bound: 0.0007978
time: 0.84 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005077, 0.0087532, -0.0006698, 0.0085953, -0.0085387, 0.0088576
1: -0.0035726, 0.0013898, -0.0036000, 0.0012982, -0.0047345, 0.0048363
2: 0.0082893, 0.0167834, 0.0084613, 0.0167772, -0.0084879, 0.0083221
3: 1.0059879, 1.0071290, 1.0059929, 1.0071430, -0.0011551, 0.0011361
4: -0.0043712, -0.0016617, -0.0043705, -0.0017180, -0.0026532, 0.0027088
5: 0.0035928, 0.0146464, 0.0034685, 0.0144438, -0.0104054, 0.0107323
6: -0.0108744, -0.0025397, -0.0107042, -0.0025416, -0.0083328, 0.0081645
7: -0.0166883, -0.0104289, -0.0165982, -0.0101711, -0.0064536, 0.0061068
8: -0.0149728, -0.0085126, -0.0149665, -0.0086314, -0.0063414, 0.0064539
9: -0.0040545, 0.0032390, -0.0038993, 0.0032361, -0.0072907, 0.0071383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0007978
time: 0.90 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0007978
time: 0.87 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005026, 0.0087532, -0.0006596, 0.0089437, -0.0088841, 0.0088565
1: -0.0035697, 0.0013898, -0.0035939, 0.0015108, -0.0049463, 0.0048375
2: 0.0082893, 0.0167834, 0.0081186, 0.0168105, -0.0085212, 0.0086649
3: 1.0059880, 1.0071189, 1.0059661, 1.0071237, -0.0011357, 0.0011529
4: -0.0043712, -0.0016617, -0.0043762, -0.0016037, -0.0027675, 0.0027145
5: 0.0035968, 0.0146464, 0.0034765, 0.0148873, -0.0108475, 0.0107316
6: -0.0108744, -0.0025405, -0.0110755, -0.0025432, -0.0083313, 0.0085350
7: -0.0166883, -0.0104316, -0.0167340, -0.0101752, -0.0064506, 0.0062403
8: -0.0149728, -0.0085129, -0.0150302, -0.0084228, -0.0065499, 0.0065173
9: -0.0040545, 0.0032390, -0.0041862, 0.0032678, -0.0073224, 0.0074252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007978
time: 0.85 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007978
time: 0.91 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0005172, 0.0087532, -0.0088576, 0.0085496
1: -0.0036000, 0.0012982, -0.0035757, 0.0013898, -0.0048363, 0.0047383
2: 0.0084613, 0.0167772, 0.0082893, 0.0167834, -0.0083221, 0.0084879
3: 1.0059929, 1.0071430, 1.0059873, 1.0071342, -0.0011413, 0.0011557
4: -0.0043705, -0.0017180, -0.0043712, -0.0016617, -0.0027088, 0.0026532
5: 0.0034685, 0.0144438, 0.0035854, 0.0146464, -0.0107323, 0.0104139
6: -0.0107042, -0.0025416, -0.0108744, -0.0025394, -0.0081648, 0.0083328
7: -0.0165982, -0.0101711, -0.0166883, -0.0104205, -0.0061152, 0.0064536
8: -0.0149665, -0.0086314, -0.0149728, -0.0085120, -0.0064546, 0.0063414
9: -0.0038993, 0.0032361, -0.0040546, 0.0032390, -0.0071383, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007982
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007975
time: 0.89 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006974, 0.0085953, -0.0085905, 0.0086219
1: -0.0036000, 0.0012982, -0.0036084, 0.0012982, -0.0047324, 0.0047441
2: 0.0084613, 0.0167772, 0.0084613, 0.0167772, -0.0083159, 0.0083159
3: 1.0059929, 1.0071430, 1.0059916, 1.0071584, -0.0011655, 0.0011514
4: -0.0043705, -0.0017180, -0.0043705, -0.0017180, -0.0026525, 0.0026525
5: 0.0034685, 0.0144438, 0.0034470, 0.0144438, -0.0104438, 0.0104683
6: -0.0107042, -0.0025416, -0.0107042, -0.0025408, -0.0081634, 0.0081626
7: -0.0165982, -0.0101711, -0.0165982, -0.0101471, -0.0063771, 0.0063526
8: -0.0149665, -0.0086314, -0.0149665, -0.0086243, -0.0063422, 0.0063351
9: -0.0038993, 0.0032361, -0.0038994, 0.0032361, -0.0071355, 0.0071356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007982
time: 1.14 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007982
time: 0.92 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0005172, 0.0087532, -0.0088565, 0.0088956
1: -0.0035939, 0.0015108, -0.0035757, 0.0013898, -0.0048375, 0.0049515
2: 0.0081186, 0.0168105, 0.0082893, 0.0167834, -0.0086649, 0.0085212
3: 1.0059661, 1.0071237, 1.0059873, 1.0071342, -0.0011681, 0.0011364
4: -0.0043762, -0.0016037, -0.0043712, -0.0016617, -0.0027145, 0.0027675
5: 0.0034765, 0.0148873, 0.0035854, 0.0146464, -0.0107315, 0.0108564
6: -0.0110755, -0.0025432, -0.0108744, -0.0025394, -0.0085361, 0.0083313
7: -0.0167340, -0.0101752, -0.0166883, -0.0104205, -0.0062512, 0.0064506
8: -0.0150302, -0.0084228, -0.0149728, -0.0085120, -0.0065182, 0.0065499
9: -0.0041862, 0.0032678, -0.0040546, 0.0032390, -0.0074252, 0.0073224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007975
time: 0.90 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007978
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006974, 0.0085953, -0.0085887, 0.0089725
1: -0.0035939, 0.0015108, -0.0036084, 0.0012982, -0.0047321, 0.0049605
2: 0.0081186, 0.0168105, 0.0084613, 0.0167772, -0.0086587, 0.0083492
3: 1.0059661, 1.0071237, 1.0059916, 1.0071584, -0.0011923, 0.0011321
4: -0.0043762, -0.0016037, -0.0043705, -0.0017180, -0.0026582, 0.0027668
5: 0.0034765, 0.0148873, 0.0034470, 0.0144438, -0.0104425, 0.0109135
6: -0.0110755, -0.0025432, -0.0107042, -0.0025408, -0.0085347, 0.0081611
7: -0.0167340, -0.0101752, -0.0165982, -0.0101471, -0.0065138, 0.0063497
8: -0.0150302, -0.0084228, -0.0149665, -0.0086243, -0.0064058, 0.0065437
9: -0.0041862, 0.0032678, -0.0038994, 0.0032361, -0.0074223, 0.0071673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007978
time: 1.06 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007978
time: 1.05 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005077, 0.0087532, -0.0006698, 0.0085953, -0.0085387, 0.0088576
1: -0.0035726, 0.0013898, -0.0036000, 0.0012982, -0.0047345, 0.0048363
2: 0.0082893, 0.0167834, 0.0084613, 0.0167772, -0.0084879, 0.0083221
3: 1.0059879, 1.0071290, 1.0059929, 1.0071430, -0.0011551, 0.0011361
4: -0.0043712, -0.0016617, -0.0043705, -0.0017180, -0.0026532, 0.0027088
5: 0.0035928, 0.0146464, 0.0034685, 0.0144438, -0.0104054, 0.0107323
6: -0.0108744, -0.0025397, -0.0107042, -0.0025416, -0.0083328, 0.0081645
7: -0.0166883, -0.0104289, -0.0165982, -0.0101711, -0.0064536, 0.0061068
8: -0.0149728, -0.0085126, -0.0149665, -0.0086314, -0.0063414, 0.0064539
9: -0.0040545, 0.0032390, -0.0038993, 0.0032361, -0.0072907, 0.0071383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005026, 0.0087532, -0.0006596, 0.0089437, -0.0088841, 0.0088565
1: -0.0035697, 0.0013898, -0.0035939, 0.0015108, -0.0049463, 0.0048375
2: 0.0082893, 0.0167834, 0.0081186, 0.0168105, -0.0085212, 0.0086649
3: 1.0059880, 1.0071189, 1.0059661, 1.0071237, -0.0011357, 0.0011529
4: -0.0043712, -0.0016617, -0.0043762, -0.0016037, -0.0027675, 0.0027145
5: 0.0035968, 0.0146464, 0.0034765, 0.0148873, -0.0108475, 0.0107316
6: -0.0108744, -0.0025405, -0.0110755, -0.0025432, -0.0083313, 0.0085350
7: -0.0166883, -0.0104316, -0.0167340, -0.0101752, -0.0064506, 0.0062403
8: -0.0149728, -0.0085129, -0.0150302, -0.0084228, -0.0065499, 0.0065173
9: -0.0040545, 0.0032390, -0.0041862, 0.0032678, -0.0073224, 0.0074252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 0.99 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 0.85 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006698, 0.0085953, -0.0085905, 0.0085905
1: -0.0036000, 0.0012982, -0.0036000, 0.0012982, -0.0047324, 0.0047324
2: 0.0084613, 0.0167772, 0.0084613, 0.0167772, -0.0083159, 0.0083159
3: 1.0059929, 1.0071430, 1.0059929, 1.0071430, -0.0011501, 0.0011501
4: -0.0043705, -0.0017180, -0.0043705, -0.0017180, -0.0026525, 0.0026525
5: 0.0034685, 0.0144438, 0.0034685, 0.0144438, -0.0104438, 0.0104438
6: -0.0107042, -0.0025416, -0.0107042, -0.0025416, -0.0081626, 0.0081626
7: -0.0165982, -0.0101711, -0.0165982, -0.0101711, -0.0063526, 0.0063526
8: -0.0149665, -0.0086314, -0.0149665, -0.0086314, -0.0063351, 0.0063351
9: -0.0038993, 0.0032361, -0.0038993, 0.0032361, -0.0071355, 0.0071355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 1.00 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
time: 1.04 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006698, 0.0085953, -0.0085923, 0.0089411
1: -0.0035939, 0.0015108, -0.0036000, 0.0012982, -0.0047332, 0.0049488
2: 0.0081186, 0.0168105, 0.0084613, 0.0167772, -0.0086587, 0.0083492
3: 1.0059661, 1.0071237, 1.0059929, 1.0071430, -0.0011770, 0.0011308
4: -0.0043762, -0.0016037, -0.0043705, -0.0017180, -0.0026582, 0.0027668
5: 0.0034765, 0.0148873, 0.0034685, 0.0144438, -0.0104457, 0.0108891
6: -0.0110755, -0.0025432, -0.0107042, -0.0025416, -0.0085339, 0.0081611
7: -0.0167340, -0.0101752, -0.0165982, -0.0101711, -0.0064893, 0.0063495
8: -0.0150302, -0.0084228, -0.0149665, -0.0086314, -0.0063988, 0.0065437
9: -0.0041862, 0.0032678, -0.0038993, 0.0032361, -0.0074223, 0.0071672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 1.02 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005077, 0.0087532, -0.0006698, 0.0085953, -0.0085387, 0.0088576
1: -0.0035726, 0.0013898, -0.0036000, 0.0012982, -0.0047345, 0.0048363
2: 0.0082893, 0.0167834, 0.0084613, 0.0167772, -0.0084879, 0.0083221
3: 1.0059879, 1.0071290, 1.0059929, 1.0071430, -0.0011551, 0.0011361
4: -0.0043712, -0.0016617, -0.0043705, -0.0017180, -0.0026532, 0.0027088
5: 0.0035928, 0.0146464, 0.0034685, 0.0144438, -0.0104054, 0.0107323
6: -0.0108744, -0.0025397, -0.0107042, -0.0025416, -0.0083328, 0.0081645
7: -0.0166883, -0.0104289, -0.0165982, -0.0101711, -0.0064536, 0.0061068
8: -0.0149728, -0.0085126, -0.0149665, -0.0086314, -0.0063414, 0.0064539
9: -0.0040545, 0.0032390, -0.0038993, 0.0032361, -0.0072907, 0.0071383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.95 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005026, 0.0087532, -0.0006596, 0.0089437, -0.0088841, 0.0088565
1: -0.0035697, 0.0013898, -0.0035939, 0.0015108, -0.0049463, 0.0048375
2: 0.0082893, 0.0167834, 0.0081186, 0.0168105, -0.0085212, 0.0086649
3: 1.0059880, 1.0071189, 1.0059661, 1.0071237, -0.0011357, 0.0011529
4: -0.0043712, -0.0016617, -0.0043762, -0.0016037, -0.0027675, 0.0027145
5: 0.0035968, 0.0146464, 0.0034765, 0.0148873, -0.0108475, 0.0107316
6: -0.0108744, -0.0025405, -0.0110755, -0.0025432, -0.0083313, 0.0085350
7: -0.0166883, -0.0104316, -0.0167340, -0.0101752, -0.0064506, 0.0062403
8: -0.0149728, -0.0085129, -0.0150302, -0.0084228, -0.0065499, 0.0065173
9: -0.0040545, 0.0032390, -0.0041862, 0.0032678, -0.0073224, 0.0074252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0008174
time: 0.96 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006596, 0.0089437, -0.0089411, 0.0085923
1: -0.0036000, 0.0012982, -0.0035939, 0.0015108, -0.0049488, 0.0047332
2: 0.0084613, 0.0167772, 0.0081186, 0.0168105, -0.0083492, 0.0086587
3: 1.0059929, 1.0071430, 1.0059661, 1.0071237, -0.0011308, 0.0011770
4: -0.0043705, -0.0017180, -0.0043762, -0.0016037, -0.0027668, 0.0026582
5: 0.0034685, 0.0144438, 0.0034765, 0.0148873, -0.0108891, 0.0104457
6: -0.0107042, -0.0025416, -0.0110755, -0.0025432, -0.0081611, 0.0085339
7: -0.0165982, -0.0101711, -0.0167340, -0.0101752, -0.0063495, 0.0064893
8: -0.0149665, -0.0086314, -0.0150302, -0.0084228, -0.0065437, 0.0063988
9: -0.0038993, 0.0032361, -0.0041862, 0.0032678, -0.0071672, 0.0074223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007975
time: 0.92 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007975
time: 0.92 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006596, 0.0089437, -0.0089189, 0.0089189
1: -0.0035939, 0.0015108, -0.0035939, 0.0015108, -0.0049383, 0.0049383
2: 0.0081186, 0.0168105, 0.0081186, 0.0168105, -0.0086919, 0.0086919
3: 1.0059661, 1.0071237, 1.0059661, 1.0071237, -0.0011576, 0.0011576
4: -0.0043762, -0.0016037, -0.0043762, -0.0016037, -0.0027725, 0.0027725
5: 0.0034765, 0.0148873, 0.0034765, 0.0148873, -0.0108717, 0.0108717
6: -0.0110755, -0.0025432, -0.0110755, -0.0025432, -0.0085323, 0.0085323
7: -0.0167340, -0.0101752, -0.0167340, -0.0101752, -0.0064838, 0.0064838
8: -0.0150302, -0.0084228, -0.0150302, -0.0084228, -0.0066073, 0.0066073
9: -0.0041862, 0.0032678, -0.0041862, 0.0032678, -0.0074540, 0.0074540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
time: 0.95 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0005172, 0.0087532, -0.0088576, 0.0085496
1: -0.0036000, 0.0012982, -0.0035757, 0.0013898, -0.0048363, 0.0047383
2: 0.0084613, 0.0167772, 0.0082893, 0.0167834, -0.0083221, 0.0084879
3: 1.0059929, 1.0071430, 1.0059873, 1.0071342, -0.0011413, 0.0011557
4: -0.0043705, -0.0017180, -0.0043712, -0.0016617, -0.0027088, 0.0026532
5: 0.0034685, 0.0144438, 0.0035854, 0.0146464, -0.0107323, 0.0104139
6: -0.0107042, -0.0025416, -0.0108744, -0.0025394, -0.0081648, 0.0083328
7: -0.0165982, -0.0101711, -0.0166883, -0.0104205, -0.0061152, 0.0064536
8: -0.0149665, -0.0086314, -0.0149728, -0.0085120, -0.0064546, 0.0063414
9: -0.0038993, 0.0032361, -0.0040546, 0.0032390, -0.0071383, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 1.06 seconds

## Relational analysis of IS_B1_A1_A2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.89 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006974, 0.0085953, -0.0085905, 0.0086219
1: -0.0036000, 0.0012982, -0.0036084, 0.0012982, -0.0047324, 0.0047441
2: 0.0084613, 0.0167772, 0.0084613, 0.0167772, -0.0083159, 0.0083159
3: 1.0059929, 1.0071430, 1.0059916, 1.0071584, -0.0011655, 0.0011514
4: -0.0043705, -0.0017180, -0.0043705, -0.0017180, -0.0026525, 0.0026525
5: 0.0034685, 0.0144438, 0.0034470, 0.0144438, -0.0104438, 0.0104683
6: -0.0107042, -0.0025416, -0.0107042, -0.0025408, -0.0081634, 0.0081626
7: -0.0165982, -0.0101711, -0.0165982, -0.0101471, -0.0063771, 0.0063526
8: -0.0149665, -0.0086314, -0.0149665, -0.0086243, -0.0063422, 0.0063351
9: -0.0038993, 0.0032361, -0.0038994, 0.0032361, -0.0071355, 0.0071356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 0.87 seconds

## Relational analysis of IS_B1_A1_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 0.87 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0005172, 0.0087532, -0.0088565, 0.0088956
1: -0.0035939, 0.0015108, -0.0035757, 0.0013898, -0.0048375, 0.0049515
2: 0.0081186, 0.0168105, 0.0082893, 0.0167834, -0.0086649, 0.0085212
3: 1.0059661, 1.0071237, 1.0059873, 1.0071342, -0.0011681, 0.0011364
4: -0.0043762, -0.0016037, -0.0043712, -0.0016617, -0.0027145, 0.0027675
5: 0.0034765, 0.0148873, 0.0035854, 0.0146464, -0.0107315, 0.0108564
6: -0.0110755, -0.0025432, -0.0108744, -0.0025394, -0.0085361, 0.0083313
7: -0.0167340, -0.0101752, -0.0166883, -0.0104205, -0.0062512, 0.0064506
8: -0.0150302, -0.0084228, -0.0149728, -0.0085120, -0.0065182, 0.0065499
9: -0.0041862, 0.0032678, -0.0040546, 0.0032390, -0.0074252, 0.0073224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.92 seconds

## Relational analysis of IS_B1_A1_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.92 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006974, 0.0085953, -0.0085887, 0.0089725
1: -0.0035939, 0.0015108, -0.0036084, 0.0012982, -0.0047321, 0.0049605
2: 0.0081186, 0.0168105, 0.0084613, 0.0167772, -0.0086587, 0.0083492
3: 1.0059661, 1.0071237, 1.0059916, 1.0071584, -0.0011923, 0.0011321
4: -0.0043762, -0.0016037, -0.0043705, -0.0017180, -0.0026582, 0.0027668
5: 0.0034765, 0.0148873, 0.0034470, 0.0144438, -0.0104425, 0.0109135
6: -0.0110755, -0.0025432, -0.0107042, -0.0025408, -0.0085347, 0.0081611
7: -0.0167340, -0.0101752, -0.0165982, -0.0101471, -0.0065138, 0.0063497
8: -0.0150302, -0.0084228, -0.0149665, -0.0086243, -0.0064058, 0.0065437
9: -0.0041862, 0.0032678, -0.0038994, 0.0032361, -0.0074223, 0.0071673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007974
time: 0.95 seconds

## Relational analysis of IS_B1_A1_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 1.09 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004880, 0.0087532, -0.0006698, 0.0085953, -0.0085163, 0.0088576
1: -0.0035666, 0.0013898, -0.0036000, 0.0012982, -0.0047270, 0.0048363
2: 0.0082894, 0.0167834, 0.0084613, 0.0167772, -0.0084879, 0.0083221
3: 1.0059888, 1.0071185, 1.0059929, 1.0071430, -0.0011542, 0.0011256
4: -0.0043712, -0.0016617, -0.0043705, -0.0017180, -0.0026532, 0.0027088
5: 0.0036080, 0.0146464, 0.0034685, 0.0144438, -0.0103880, 0.0107323
6: -0.0108744, -0.0025402, -0.0107042, -0.0025416, -0.0083328, 0.0081641
7: -0.0166883, -0.0104462, -0.0165982, -0.0101711, -0.0064535, 0.0060895
8: -0.0149728, -0.0085139, -0.0149665, -0.0086314, -0.0063414, 0.0064526
9: -0.0040545, 0.0032390, -0.0038993, 0.0032361, -0.0072906, 0.0071383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.87 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 1.03 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006698, 0.0085953, -0.0085905, 0.0085905
1: -0.0036000, 0.0012982, -0.0036000, 0.0012982, -0.0047324, 0.0047324
2: 0.0084613, 0.0167772, 0.0084613, 0.0167772, -0.0083159, 0.0083159
3: 1.0059929, 1.0071430, 1.0059929, 1.0071430, -0.0011501, 0.0011501
4: -0.0043705, -0.0017180, -0.0043705, -0.0017180, -0.0026525, 0.0026525
5: 0.0034685, 0.0144438, 0.0034685, 0.0144438, -0.0104438, 0.0104438
6: -0.0107042, -0.0025416, -0.0107042, -0.0025416, -0.0081626, 0.0081626
7: -0.0165982, -0.0101711, -0.0165982, -0.0101711, -0.0063526, 0.0063526
8: -0.0149665, -0.0086314, -0.0149665, -0.0086314, -0.0063351, 0.0063351
9: -0.0038993, 0.0032361, -0.0038993, 0.0032361, -0.0071355, 0.0071355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 1.16 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.96 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0004833, 0.0091788, -0.0092902, 0.0085235
1: -0.0036000, 0.0012982, -0.0035619, 0.0016502, -0.0051033, 0.0047267
2: 0.0084613, 0.0167772, 0.0078704, 0.0168235, -0.0083622, 0.0089068
3: 1.0059929, 1.0071430, 1.0059547, 1.0070997, -0.0011069, 0.0011883
4: -0.0043705, -0.0017180, -0.0043777, -0.0015219, -0.0028486, 0.0026597
5: 0.0034685, 0.0144438, 0.0036119, 0.0151884, -0.0112799, 0.0103938
6: -0.0107042, -0.0025416, -0.0113282, -0.0025419, -0.0081624, 0.0087866
7: -0.0165982, -0.0101711, -0.0168552, -0.0104462, -0.0060900, 0.0066215
8: -0.0149665, -0.0086314, -0.0150438, -0.0082550, -0.0067115, 0.0064124
9: -0.0038993, 0.0032361, -0.0044062, 0.0032751, -0.0071744, 0.0076424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.95 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.97 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006596, 0.0089437, -0.0089411, 0.0085923
1: -0.0036000, 0.0012982, -0.0035939, 0.0015108, -0.0049488, 0.0047332
2: 0.0084613, 0.0167772, 0.0081186, 0.0168105, -0.0083492, 0.0086587
3: 1.0059929, 1.0071430, 1.0059661, 1.0071237, -0.0011308, 0.0011770
4: -0.0043705, -0.0017180, -0.0043762, -0.0016037, -0.0027668, 0.0026582
5: 0.0034685, 0.0144438, 0.0034765, 0.0148873, -0.0108891, 0.0104457
6: -0.0107042, -0.0025416, -0.0110755, -0.0025432, -0.0081611, 0.0085339
7: -0.0165982, -0.0101711, -0.0167340, -0.0101752, -0.0063495, 0.0064893
8: -0.0149665, -0.0086314, -0.0150302, -0.0084228, -0.0065437, 0.0063988
9: -0.0038993, 0.0032361, -0.0041862, 0.0032678, -0.0071672, 0.0074223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
time: 1.03 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0005172, 0.0087532, -0.0088576, 0.0085496
1: -0.0036000, 0.0012982, -0.0035757, 0.0013898, -0.0048363, 0.0047383
2: 0.0084613, 0.0167772, 0.0082893, 0.0167834, -0.0083221, 0.0084879
3: 1.0059929, 1.0071430, 1.0059873, 1.0071342, -0.0011413, 0.0011557
4: -0.0043705, -0.0017180, -0.0043712, -0.0016617, -0.0027088, 0.0026532
5: 0.0034685, 0.0144438, 0.0035854, 0.0146464, -0.0107323, 0.0104139
6: -0.0107042, -0.0025416, -0.0108744, -0.0025394, -0.0081648, 0.0083328
7: -0.0165982, -0.0101711, -0.0166883, -0.0104205, -0.0061152, 0.0064536
8: -0.0149665, -0.0086314, -0.0149728, -0.0085120, -0.0064546, 0.0063414
9: -0.0038993, 0.0032361, -0.0040546, 0.0032390, -0.0071383, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 1.03 seconds

## Relational analysis of IS_B1_A1_A2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.88 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006698, 0.0085953, -0.0006974, 0.0085953, -0.0085905, 0.0086219
1: -0.0036000, 0.0012982, -0.0036084, 0.0012982, -0.0047324, 0.0047441
2: 0.0084613, 0.0167772, 0.0084613, 0.0167772, -0.0083159, 0.0083159
3: 1.0059929, 1.0071430, 1.0059916, 1.0071584, -0.0011655, 0.0011514
4: -0.0043705, -0.0017180, -0.0043705, -0.0017180, -0.0026525, 0.0026525
5: 0.0034685, 0.0144438, 0.0034470, 0.0144438, -0.0104438, 0.0104683
6: -0.0107042, -0.0025416, -0.0107042, -0.0025408, -0.0081634, 0.0081626
7: -0.0165982, -0.0101711, -0.0165982, -0.0101471, -0.0063771, 0.0063526
8: -0.0149665, -0.0086314, -0.0149665, -0.0086243, -0.0063422, 0.0063351
9: -0.0038993, 0.0032361, -0.0038994, 0.0032361, -0.0071355, 0.0071356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 0.90 seconds

## Relational analysis of IS_B1_A1_A2_A2_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
time: 0.90 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0005172, 0.0087532, -0.0088565, 0.0088956
1: -0.0035939, 0.0015108, -0.0035757, 0.0013898, -0.0048375, 0.0049515
2: 0.0081186, 0.0168105, 0.0082893, 0.0167834, -0.0086649, 0.0085212
3: 1.0059661, 1.0071237, 1.0059873, 1.0071342, -0.0011681, 0.0011364
4: -0.0043762, -0.0016037, -0.0043712, -0.0016617, -0.0027145, 0.0027675
5: 0.0034765, 0.0148873, 0.0035854, 0.0146464, -0.0107315, 0.0108564
6: -0.0110755, -0.0025432, -0.0108744, -0.0025394, -0.0085361, 0.0083313
7: -0.0167340, -0.0101752, -0.0166883, -0.0104205, -0.0062512, 0.0064506
8: -0.0150302, -0.0084228, -0.0149728, -0.0085120, -0.0065182, 0.0065499
9: -0.0041862, 0.0032678, -0.0040546, 0.0032390, -0.0074252, 0.0073224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
time: 0.96 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006974, 0.0085953, -0.0085887, 0.0089725
1: -0.0035939, 0.0015108, -0.0036084, 0.0012982, -0.0047321, 0.0049605
2: 0.0081186, 0.0168105, 0.0084613, 0.0167772, -0.0086587, 0.0083492
3: 1.0059661, 1.0071237, 1.0059916, 1.0071584, -0.0011923, 0.0011321
4: -0.0043762, -0.0016037, -0.0043705, -0.0017180, -0.0026582, 0.0027668
5: 0.0034765, 0.0148873, 0.0034470, 0.0144438, -0.0104425, 0.0109135
6: -0.0110755, -0.0025432, -0.0107042, -0.0025408, -0.0085347, 0.0081611
7: -0.0167340, -0.0101752, -0.0165982, -0.0101471, -0.0065138, 0.0063497
8: -0.0150302, -0.0084228, -0.0149665, -0.0086243, -0.0064058, 0.0065437
9: -0.0041862, 0.0032678, -0.0038994, 0.0032361, -0.0074223, 0.0071673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
time: 0.93 seconds

## Relational analysis of IS_B1_A1_A2_A2_B1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
time: 0.90 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004833, 0.0091788, -0.0006698, 0.0085953, -0.0085235, 0.0092902
1: -0.0035619, 0.0016502, -0.0036000, 0.0012982, -0.0047267, 0.0051033
2: 0.0078704, 0.0168235, 0.0084613, 0.0167772, -0.0089068, 0.0083622
3: 1.0059547, 1.0070997, 1.0059929, 1.0071430, -0.0011883, 0.0011069
4: -0.0043777, -0.0015219, -0.0043705, -0.0017180, -0.0026597, 0.0028486
5: 0.0036119, 0.0151884, 0.0034685, 0.0144438, -0.0103938, 0.0112799
6: -0.0113282, -0.0025419, -0.0107042, -0.0025416, -0.0087866, 0.0081624
7: -0.0168552, -0.0104462, -0.0165982, -0.0101711, -0.0066215, 0.0060900
8: -0.0150438, -0.0082550, -0.0149665, -0.0086314, -0.0064124, 0.0067115
9: -0.0044062, 0.0032751, -0.0038993, 0.0032361, -0.0076424, 0.0071744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 0.86 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006698, 0.0085953, -0.0085923, 0.0089411
1: -0.0035939, 0.0015108, -0.0036000, 0.0012982, -0.0047332, 0.0049488
2: 0.0081186, 0.0168105, 0.0084613, 0.0167772, -0.0086587, 0.0083492
3: 1.0059661, 1.0071237, 1.0059929, 1.0071430, -0.0011770, 0.0011308
4: -0.0043762, -0.0016037, -0.0043705, -0.0017180, -0.0026582, 0.0027668
5: 0.0034765, 0.0148873, 0.0034685, 0.0144438, -0.0104457, 0.0108891
6: -0.0110755, -0.0025432, -0.0107042, -0.0025416, -0.0085339, 0.0081611
7: -0.0167340, -0.0101752, -0.0165982, -0.0101711, -0.0064893, 0.0063495
8: -0.0150302, -0.0084228, -0.0149665, -0.0086314, -0.0063988, 0.0065437
9: -0.0041862, 0.0032678, -0.0038993, 0.0032361, -0.0074223, 0.0071672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 1.17 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004833, 0.0091788, -0.0006596, 0.0089437, -0.0088427, 0.0092622
1: -0.0035619, 0.0016502, -0.0035939, 0.0015108, -0.0049308, 0.0050910
2: 0.0078704, 0.0168235, 0.0081186, 0.0168105, -0.0089401, 0.0087049
3: 1.0059547, 1.0070997, 1.0059661, 1.0071237, -0.0011690, 0.0011337
4: -0.0043777, -0.0015219, -0.0043762, -0.0016037, -0.0027740, 0.0028543
5: 0.0036119, 0.0151884, 0.0034765, 0.0148873, -0.0108152, 0.0112580
6: -0.0113282, -0.0025419, -0.0110755, -0.0025432, -0.0087850, 0.0085336
7: -0.0168552, -0.0104462, -0.0167340, -0.0101752, -0.0066154, 0.0062233
8: -0.0150438, -0.0082550, -0.0150302, -0.0084228, -0.0066209, 0.0067751
9: -0.0044062, 0.0032751, -0.0041862, 0.0032678, -0.0076741, 0.0074613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: A, layer: 3, pos: 237

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006596, 0.0089437, -0.0006596, 0.0089437, -0.0089189, 0.0089189
1: -0.0035939, 0.0015108, -0.0035939, 0.0015108, -0.0049383, 0.0049383
2: 0.0081186, 0.0168105, 0.0081186, 0.0168105, -0.0086919, 0.0086919
3: 1.0059661, 1.0071237, 1.0059661, 1.0071237, -0.0011576, 0.0011576
4: -0.0043762, -0.0016037, -0.0043762, -0.0016037, -0.0027725, 0.0027725
5: 0.0034765, 0.0148873, 0.0034765, 0.0148873, -0.0108717, 0.0108717
6: -0.0110755, -0.0025432, -0.0110755, -0.0025432, -0.0085323, 0.0085323
7: -0.0167340, -0.0101752, -0.0167340, -0.0101752, -0.0064838, 0.0064838
8: -0.0150302, -0.0084228, -0.0150302, -0.0084228, -0.0066073, 0.0066073
9: -0.0041862, 0.0032678, -0.0041862, 0.0032678, -0.0074540, 0.0074540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: B, layer: 3, pos: 64
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A1_A2_A2_B2_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
time: 1.24 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
time: 1.33 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.83 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.80 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.95 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.08 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006401, 0.0081766, -0.0005172, 0.0087532, -0.0088675, 0.0081392
1: -0.0035629, 0.0010552, -0.0035757, 0.0013898, -0.0048281, 0.0044937
2: 0.0089210, 0.0167706, 0.0082893, 0.0167834, -0.0078624, 0.0084813
3: 1.0059268, 1.0070760, 1.0059873, 1.0071342, -0.0012074, 0.0010887
4: -0.0043726, -0.0018687, -0.0043712, -0.0016617, -0.0027109, 0.0025025
5: 0.0034946, 0.0139053, 0.0035854, 0.0146464, -0.0107368, 0.0098828
6: -0.0102517, -0.0025453, -0.0108744, -0.0025394, -0.0077123, 0.0083292
7: -0.0163553, -0.0099628, -0.0166883, -0.0104205, -0.0058753, 0.0066686
8: -0.0150165, -0.0089727, -0.0149728, -0.0085120, -0.0065045, 0.0060001
9: -0.0034833, 0.0032784, -0.0040546, 0.0032390, -0.0067223, 0.0073330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.81 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.86 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006399, 0.0077967, -0.0006974, 0.0085953, -0.0087225, 0.0079487
1: -0.0035629, 0.0008257, -0.0036084, 0.0012982, -0.0047399, 0.0042809
2: 0.0093030, 0.0167430, 0.0084613, 0.0167772, -0.0074742, 0.0082816
3: 1.0059569, 1.0070760, 1.0059916, 1.0071584, -0.0012015, 0.0010844
4: -0.0043683, -0.0019954, -0.0043705, -0.0017180, -0.0026503, 0.0023751
5: 0.0034948, 0.0134210, 0.0034470, 0.0144438, -0.0105449, 0.0095437
6: -0.0098461, -0.0025465, -0.0107042, -0.0025408, -0.0073053, 0.0081578
7: -0.0161838, -0.0099628, -0.0165982, -0.0101471, -0.0059765, 0.0065804
8: -0.0149672, -0.0092253, -0.0149665, -0.0086243, -0.0063429, 0.0057412
9: -0.0031504, 0.0032561, -0.0038994, 0.0032361, -0.0063865, 0.0071556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.10 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.27 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.89 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.85 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.93 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.03 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006401, 0.0081766, -0.0005172, 0.0087532, -0.0088675, 0.0081392
1: -0.0035629, 0.0010552, -0.0035757, 0.0013898, -0.0048281, 0.0044937
2: 0.0089210, 0.0167706, 0.0082893, 0.0167834, -0.0078624, 0.0084813
3: 1.0059268, 1.0070760, 1.0059873, 1.0071342, -0.0012074, 0.0010887
4: -0.0043726, -0.0018687, -0.0043712, -0.0016617, -0.0027109, 0.0025025
5: 0.0034946, 0.0139053, 0.0035854, 0.0146464, -0.0107368, 0.0098828
6: -0.0102517, -0.0025453, -0.0108744, -0.0025394, -0.0077123, 0.0083292
7: -0.0163553, -0.0099628, -0.0166883, -0.0104205, -0.0058753, 0.0066686
8: -0.0150165, -0.0089727, -0.0149728, -0.0085120, -0.0065045, 0.0060001
9: -0.0034833, 0.0032784, -0.0040546, 0.0032390, -0.0067223, 0.0073330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.85 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.84 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006399, 0.0077967, -0.0006974, 0.0085953, -0.0087225, 0.0079487
1: -0.0035629, 0.0008257, -0.0036084, 0.0012982, -0.0047399, 0.0042809
2: 0.0093030, 0.0167430, 0.0084613, 0.0167772, -0.0074742, 0.0082816
3: 1.0059569, 1.0070760, 1.0059916, 1.0071584, -0.0012015, 0.0010844
4: -0.0043683, -0.0019954, -0.0043705, -0.0017180, -0.0026503, 0.0023751
5: 0.0034948, 0.0134210, 0.0034470, 0.0144438, -0.0105449, 0.0095437
6: -0.0098461, -0.0025465, -0.0107042, -0.0025408, -0.0073053, 0.0081578
7: -0.0161838, -0.0099628, -0.0165982, -0.0101471, -0.0059765, 0.0065804
8: -0.0149672, -0.0092253, -0.0149665, -0.0086243, -0.0063429, 0.0057412
9: -0.0031504, 0.0032561, -0.0038994, 0.0032361, -0.0063865, 0.0071556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.16 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.84 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.94 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.09 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006401, 0.0081766, -0.0005172, 0.0087532, -0.0088675, 0.0081392
1: -0.0035629, 0.0010552, -0.0035757, 0.0013898, -0.0048281, 0.0044937
2: 0.0089210, 0.0167706, 0.0082893, 0.0167834, -0.0078624, 0.0084813
3: 1.0059268, 1.0070760, 1.0059873, 1.0071342, -0.0012074, 0.0010887
4: -0.0043726, -0.0018687, -0.0043712, -0.0016617, -0.0027109, 0.0025025
5: 0.0034946, 0.0139053, 0.0035854, 0.0146464, -0.0107368, 0.0098828
6: -0.0102517, -0.0025453, -0.0108744, -0.0025394, -0.0077123, 0.0083292
7: -0.0163553, -0.0099628, -0.0166883, -0.0104205, -0.0058753, 0.0066686
8: -0.0150165, -0.0089727, -0.0149728, -0.0085120, -0.0065045, 0.0060001
9: -0.0034833, 0.0032784, -0.0040546, 0.0032390, -0.0067223, 0.0073330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.82 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.86 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006399, 0.0077967, -0.0006974, 0.0085953, -0.0087225, 0.0079487
1: -0.0035629, 0.0008257, -0.0036084, 0.0012982, -0.0047399, 0.0042809
2: 0.0093030, 0.0167430, 0.0084613, 0.0167772, -0.0074742, 0.0082816
3: 1.0059569, 1.0070760, 1.0059916, 1.0071584, -0.0012015, 0.0010844
4: -0.0043683, -0.0019954, -0.0043705, -0.0017180, -0.0026503, 0.0023751
5: 0.0034948, 0.0134210, 0.0034470, 0.0144438, -0.0105449, 0.0095437
6: -0.0098461, -0.0025465, -0.0107042, -0.0025408, -0.0073053, 0.0081578
7: -0.0161838, -0.0099628, -0.0165982, -0.0101471, -0.0059765, 0.0065804
8: -0.0149672, -0.0092253, -0.0149665, -0.0086243, -0.0063429, 0.0057412
9: -0.0031504, 0.0032561, -0.0038994, 0.0032361, -0.0063865, 0.0071556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.07 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.88 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.86 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.89 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.03 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006401, 0.0081766, -0.0005172, 0.0087532, -0.0088675, 0.0081392
1: -0.0035629, 0.0010552, -0.0035757, 0.0013898, -0.0048281, 0.0044937
2: 0.0089210, 0.0167706, 0.0082893, 0.0167834, -0.0078624, 0.0084813
3: 1.0059268, 1.0070760, 1.0059873, 1.0071342, -0.0012074, 0.0010887
4: -0.0043726, -0.0018687, -0.0043712, -0.0016617, -0.0027109, 0.0025025
5: 0.0034946, 0.0139053, 0.0035854, 0.0146464, -0.0107368, 0.0098828
6: -0.0102517, -0.0025453, -0.0108744, -0.0025394, -0.0077123, 0.0083292
7: -0.0163553, -0.0099628, -0.0166883, -0.0104205, -0.0058753, 0.0066686
8: -0.0150165, -0.0089727, -0.0149728, -0.0085120, -0.0065045, 0.0060001
9: -0.0034833, 0.0032784, -0.0040546, 0.0032390, -0.0067223, 0.0073330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.83 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.83 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006399, 0.0077967, -0.0006974, 0.0085953, -0.0087225, 0.0079487
1: -0.0035629, 0.0008257, -0.0036084, 0.0012982, -0.0047399, 0.0042809
2: 0.0093030, 0.0167430, 0.0084613, 0.0167772, -0.0074742, 0.0082816
3: 1.0059569, 1.0070760, 1.0059916, 1.0071584, -0.0012015, 0.0010844
4: -0.0043683, -0.0019954, -0.0043705, -0.0017180, -0.0026503, 0.0023751
5: 0.0034948, 0.0134210, 0.0034470, 0.0144438, -0.0105449, 0.0095437
6: -0.0098461, -0.0025465, -0.0107042, -0.0025408, -0.0073053, 0.0081578
7: -0.0161838, -0.0099628, -0.0165982, -0.0101471, -0.0059765, 0.0065804
8: -0.0149672, -0.0092253, -0.0149665, -0.0086243, -0.0063429, 0.0057412
9: -0.0031504, 0.0032561, -0.0038994, 0.0032361, -0.0063865, 0.0071556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.02 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.88 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.86 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.89 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.03 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006401, 0.0081766, -0.0005172, 0.0087532, -0.0088675, 0.0081392
1: -0.0035629, 0.0010552, -0.0035757, 0.0013898, -0.0048281, 0.0044937
2: 0.0089210, 0.0167706, 0.0082893, 0.0167834, -0.0078624, 0.0084813
3: 1.0059268, 1.0070760, 1.0059873, 1.0071342, -0.0012074, 0.0010887
4: -0.0043726, -0.0018687, -0.0043712, -0.0016617, -0.0027109, 0.0025025
5: 0.0034946, 0.0139053, 0.0035854, 0.0146464, -0.0107368, 0.0098828
6: -0.0102517, -0.0025453, -0.0108744, -0.0025394, -0.0077123, 0.0083292
7: -0.0163553, -0.0099628, -0.0166883, -0.0104205, -0.0058753, 0.0066686
8: -0.0150165, -0.0089727, -0.0149728, -0.0085120, -0.0065045, 0.0060001
9: -0.0034833, 0.0032784, -0.0040546, 0.0032390, -0.0067223, 0.0073330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.84 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.84 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006399, 0.0077967, -0.0006974, 0.0085953, -0.0087225, 0.0079487
1: -0.0035629, 0.0008257, -0.0036084, 0.0012982, -0.0047399, 0.0042809
2: 0.0093030, 0.0167430, 0.0084613, 0.0167772, -0.0074742, 0.0082816
3: 1.0059569, 1.0070760, 1.0059916, 1.0071584, -0.0012015, 0.0010844
4: -0.0043683, -0.0019954, -0.0043705, -0.0017180, -0.0026503, 0.0023751
5: 0.0034948, 0.0134210, 0.0034470, 0.0144438, -0.0105449, 0.0095437
6: -0.0098461, -0.0025465, -0.0107042, -0.0025408, -0.0073053, 0.0081578
7: -0.0161838, -0.0099628, -0.0165982, -0.0101471, -0.0059765, 0.0065804
8: -0.0149672, -0.0092253, -0.0149665, -0.0086243, -0.0063429, 0.0057412
9: -0.0031504, 0.0032561, -0.0038994, 0.0032361, -0.0063865, 0.0071556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.18 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.92 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.87 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
time: 0.96 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006401, 0.0081766, -0.0005172, 0.0087532, -0.0088675, 0.0081392
1: -0.0035629, 0.0010552, -0.0035757, 0.0013898, -0.0048281, 0.0044937
2: 0.0089210, 0.0167706, 0.0082893, 0.0167834, -0.0078624, 0.0084813
3: 1.0059268, 1.0070760, 1.0059873, 1.0071342, -0.0012074, 0.0010887
4: -0.0043726, -0.0018687, -0.0043712, -0.0016617, -0.0027109, 0.0025025
5: 0.0034946, 0.0139053, 0.0035854, 0.0146464, -0.0107368, 0.0098828
6: -0.0102517, -0.0025453, -0.0108744, -0.0025394, -0.0077123, 0.0083292
7: -0.0163553, -0.0099628, -0.0166883, -0.0104205, -0.0058753, 0.0066686
8: -0.0150165, -0.0089727, -0.0149728, -0.0085120, -0.0065045, 0.0060001
9: -0.0034833, 0.0032784, -0.0040546, 0.0032390, -0.0067223, 0.0073330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.84 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.85 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006399, 0.0077967, -0.0006974, 0.0085953, -0.0087225, 0.0079487
1: -0.0035629, 0.0008257, -0.0036084, 0.0012982, -0.0047399, 0.0042809
2: 0.0093030, 0.0167430, 0.0084613, 0.0167772, -0.0074742, 0.0082816
3: 1.0059569, 1.0070760, 1.0059916, 1.0071584, -0.0012015, 0.0010844
4: -0.0043683, -0.0019954, -0.0043705, -0.0017180, -0.0026503, 0.0023751
5: 0.0034948, 0.0134210, 0.0034470, 0.0144438, -0.0105449, 0.0095437
6: -0.0098461, -0.0025465, -0.0107042, -0.0025408, -0.0073053, 0.0081578
7: -0.0161838, -0.0099628, -0.0165982, -0.0101471, -0.0059765, 0.0065804
8: -0.0149672, -0.0092253, -0.0149665, -0.0086243, -0.0063429, 0.0057412
9: -0.0031504, 0.0032561, -0.0038994, 0.0032361, -0.0063865, 0.0071556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 1.06 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 1.03 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.86 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.83 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
time: 0.91 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006401, 0.0081766, -0.0005172, 0.0087532, -0.0088675, 0.0081392
1: -0.0035629, 0.0010552, -0.0035757, 0.0013898, -0.0048281, 0.0044937
2: 0.0089210, 0.0167706, 0.0082893, 0.0167834, -0.0078624, 0.0084813
3: 1.0059268, 1.0070760, 1.0059873, 1.0071342, -0.0012074, 0.0010887
4: -0.0043726, -0.0018687, -0.0043712, -0.0016617, -0.0027109, 0.0025025
5: 0.0034946, 0.0139053, 0.0035854, 0.0146464, -0.0107368, 0.0098828
6: -0.0102517, -0.0025453, -0.0108744, -0.0025394, -0.0077123, 0.0083292
7: -0.0163553, -0.0099628, -0.0166883, -0.0104205, -0.0058753, 0.0066686
8: -0.0150165, -0.0089727, -0.0149728, -0.0085120, -0.0065045, 0.0060001
9: -0.0034833, 0.0032784, -0.0040546, 0.0032390, -0.0067223, 0.0073330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.85 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.83 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006399, 0.0077967, -0.0006974, 0.0085953, -0.0087225, 0.0079487
1: -0.0035629, 0.0008257, -0.0036084, 0.0012982, -0.0047399, 0.0042809
2: 0.0093030, 0.0167430, 0.0084613, 0.0167772, -0.0074742, 0.0082816
3: 1.0059569, 1.0070760, 1.0059916, 1.0071584, -0.0012015, 0.0010844
4: -0.0043683, -0.0019954, -0.0043705, -0.0017180, -0.0026503, 0.0023751
5: 0.0034948, 0.0134210, 0.0034470, 0.0144438, -0.0105449, 0.0095437
6: -0.0098461, -0.0025465, -0.0107042, -0.0025408, -0.0073053, 0.0081578
7: -0.0161838, -0.0099628, -0.0165982, -0.0101471, -0.0059765, 0.0065804
8: -0.0149672, -0.0092253, -0.0149665, -0.0086243, -0.0063429, 0.0057412
9: -0.0031504, 0.0032561, -0.0038994, 0.0032361, -0.0063865, 0.0071556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 1.02 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
time: 0.89 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006442, 0.0078524, -0.0005172, 0.0087532, -0.0088648, 0.0078124
1: -0.0035670, 0.0008563, -0.0035757, 0.0013898, -0.0048269, 0.0042916
2: 0.0092383, 0.0167391, 0.0082893, 0.0167834, -0.0075451, 0.0084498
3: 1.0059520, 1.0070909, 1.0059873, 1.0071342, -0.0011822, 0.0011036
4: -0.0043671, -0.0019749, -0.0043712, -0.0016617, -0.0027054, 0.0023962
5: 0.0034914, 0.0134924, 0.0035854, 0.0146464, -0.0107346, 0.0094671
6: -0.0099061, -0.0025440, -0.0108744, -0.0025394, -0.0073666, 0.0083304
7: -0.0162294, -0.0099621, -0.0166883, -0.0104205, -0.0057483, 0.0066694
8: -0.0149536, -0.0091554, -0.0149728, -0.0085120, -0.0064416, 0.0058174
9: -0.0032160, 0.0032465, -0.0040546, 0.0032390, -0.0064550, 0.0073010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
time: 0.86 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006439, 0.0074979, -0.0006974, 0.0085953, -0.0087218, 0.0076456
1: -0.0035670, 0.0006425, -0.0036084, 0.0012982, -0.0047397, 0.0040952
2: 0.0095980, 0.0167132, 0.0084613, 0.0167772, -0.0071793, 0.0082519
3: 1.0059806, 1.0070909, 1.0059916, 1.0071584, -0.0011778, 0.0010993
4: -0.0043631, -0.0020940, -0.0043705, -0.0017180, -0.0026451, 0.0022765
5: 0.0034916, 0.0130401, 0.0034470, 0.0144438, -0.0105442, 0.0091596
6: -0.0095271, -0.0025453, -0.0107042, -0.0025408, -0.0069863, 0.0081590
7: -0.0160676, -0.0099622, -0.0165982, -0.0101471, -0.0058598, 0.0065815
8: -0.0149102, -0.0093882, -0.0149665, -0.0086243, -0.0062859, 0.0055783
9: -0.0029036, 0.0032264, -0.0038994, 0.0032361, -0.0061397, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 65
type: A, layer: 3, pos: 254
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 64

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
time: 0.97 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
time: 1.06 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.58 seconds
IS_B1_A1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008203, upper bound: 0.0008151
IS_B1_A1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008151, upper bound: 0.0008151
IS_B1_A1_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008024, upper bound: 0.0007982
IS_B1_A1_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008024, upper bound: 0.0007978
IS_B1_A1_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0007978
IS_B1_A1_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0007978
IS_B1_A1_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007978
IS_B1_A1_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007978
IS_B1_A1_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007982
IS_B1_A1_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007975
IS_B1_A1_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007982
IS_B1_A1_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007982
IS_B1_A1_A1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007975
IS_B1_A1_A1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007978
IS_B1_A1_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007978
IS_B1_A1_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0007978
IS_B1_A1_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007975
IS_B1_A1_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0007975
IS_B1_A1_A1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
IS_B1_A1_A1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
IS_B1_A1_A2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007974
IS_B1_A1_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
IS_B1_A1_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0008174
IS_B1_A1_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007982
IS_B1_A1_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007975
IS_B1_A1_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
IS_B1_A1_A2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
IS_B1_A1_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0008174, upper bound: 0.0007978
IS_B1_A1_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007982, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007974, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007978, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
IS_B1_A1_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0007975, upper bound: 0.0008174
IS_B1_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
IS_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B1_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007379
IS_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007379
IS_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006959, upper bound: 0.0007361
IS_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007482
IS_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007361
IS_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0006934, upper bound: 0.0007442
IS_B2_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
IS_B2_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006959
IS_B2_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006959
IS_B2_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007379, upper bound: 0.0006934
IS_B2_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007482, upper bound: 0.0006934
IS_B2_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007361, upper bound: 0.0006934
IS_B2_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0007442, upper bound: 0.0006934
IS_B2_A2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0006724, upper bound: 0.0006491
IS_B2_A2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 3, lower bound: -0.0006669, upper bound: 0.0006472

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.46 + 598.36 = 601.82 seconds
