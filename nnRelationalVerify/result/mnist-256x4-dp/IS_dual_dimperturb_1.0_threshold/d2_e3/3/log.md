## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00157248


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0062809, 0.0062809)
1: (-0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017708, 0.0017708)
2: (-0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0130656, 0.0130656)
3: (0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017290, 0.0017290)
4: (-0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0097644, 0.0097644)
5: (0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0027128, 0.0027128)
6: (0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024624, 0.0024624)
7: (-0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0091894, 0.0091894)
8: (-0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0071521, 0.0071521)
9: (-0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006171, 0.0006171)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 2.95 = 4.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0017475, upper bound: 0.0017472

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016955, upper bound: 0.0017030
time: 1.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0017045, upper bound: 0.0017047
time: 1.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.80
Output dim: 5, lower bound: -0.0016955, upper bound: 0.0017030
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.80
Output dim: 5, lower bound: -0.0017045, upper bound: 0.0017047

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0105328, -0.0035968, -0.0105793, -0.0034076, -0.0060218, 0.0056248
1: -0.0059083, -0.0039527, -0.0059213, -0.0038994, -0.0016978, 0.0015858
2: -0.0050326, 0.0093958, -0.0051291, 0.0097893, -0.0125265, 0.0117008
3: 0.0009613, 0.0028707, 0.0009485, 0.0029228, -0.0016577, 0.0015484
4: -0.0009300, 0.0098529, -0.0012240, 0.0099251, -0.0087444, 0.0093615
5: 0.9952478, 0.9982436, 0.9951662, 0.9982637, -0.0024295, 0.0026009
6: 0.0035701, 0.0062894, 0.0034960, 0.0063076, -0.0022052, 0.0023608
7: -0.0100583, 0.0000895, -0.0103351, 0.0001575, -0.0082295, 0.0088103
8: -0.0092626, -0.0013645, -0.0093154, -0.0011490, -0.0068570, 0.0064050
9: -0.0038920, -0.0032106, -0.0039106, -0.0032060, -0.0005526, 0.0005916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016529, upper bound: 0.0016586
time: 1.36 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016505, upper bound: 0.0016587
time: 1.96 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0105654, -0.0035198, -0.0105814, -0.0033796, -0.0061672, 0.0058830
1: -0.0059174, -0.0039310, -0.0059220, -0.0038915, -0.0017388, 0.0016586
2: -0.0051003, 0.0095560, -0.0051337, 0.0098476, -0.0128291, 0.0122379
3: 0.0009524, 0.0028919, 0.0009479, 0.0029305, -0.0016977, 0.0016195
4: -0.0010497, 0.0099035, -0.0012676, 0.0099285, -0.0091458, 0.0095877
5: 0.9952146, 0.9982577, 0.9951540, 0.9982646, -0.0025410, 0.0026637
6: 0.0035399, 0.0063022, 0.0034850, 0.0063085, -0.0023064, 0.0024179
7: -0.0101710, 0.0001372, -0.0103761, 0.0001607, -0.0086073, 0.0090231
8: -0.0092996, -0.0012767, -0.0093179, -0.0011171, -0.0070227, 0.0066990
9: -0.0038996, -0.0032074, -0.0039134, -0.0032058, -0.0005780, 0.0006059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016616, upper bound: 0.0016601
time: 1.90 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016601, upper bound: 0.0016601
time: 1.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.99 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 5, lower bound: -0.0016529, upper bound: 0.0016586
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 5, lower bound: -0.0016505, upper bound: 0.0016587
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 5, lower bound: -0.0016616, upper bound: 0.0016601
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 5, lower bound: -0.0016601, upper bound: 0.0016601

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0106038, -0.0038284, -0.0105645, -0.0034872, -0.0058608, 0.0053044
1: -0.0059283, -0.0040180, -0.0059172, -0.0039218, -0.0016524, 0.0014955
2: -0.0051802, 0.0089141, -0.0050984, 0.0096239, -0.0121916, 0.0110342
3: 0.0009418, 0.0028069, 0.0009526, 0.0029009, -0.0016134, 0.0014602
4: -0.0005700, 0.0099632, -0.0011004, 0.0099021, -0.0082462, 0.0091112
5: 0.9953479, 0.9982744, 0.9952005, 0.9982573, -0.0022910, 0.0025314
6: 0.0036609, 0.0063173, 0.0035272, 0.0063018, -0.0020796, 0.0022977
7: -0.0097195, 0.0001934, -0.0102187, 0.0001358, -0.0077606, 0.0085747
8: -0.0093434, -0.0016281, -0.0092986, -0.0012396, -0.0066737, 0.0060401
9: -0.0038693, -0.0032036, -0.0039028, -0.0032075, -0.0005211, 0.0005758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016237, upper bound: 0.0016379
time: 2.06 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016320, upper bound: 0.0016378
time: 1.90 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0105093, -0.0037568, -0.0105749, -0.0034397, -0.0059813, 0.0053506
1: -0.0059016, -0.0039978, -0.0059201, -0.0039084, -0.0016864, 0.0015085
2: -0.0049837, 0.0090629, -0.0051201, 0.0097226, -0.0124424, 0.0111304
3: 0.0009678, 0.0028266, 0.0009497, 0.0029139, -0.0016465, 0.0014729
4: -0.0006812, 0.0098164, -0.0011742, 0.0099183, -0.0083182, 0.0092986
5: 0.9953170, 0.9982336, 0.9951801, 0.9982618, -0.0023110, 0.0025834
6: 0.0036329, 0.0062802, 0.0035086, 0.0063059, -0.0020977, 0.0023450
7: -0.0098242, 0.0000552, -0.0102882, 0.0001511, -0.0078283, 0.0087511
8: -0.0092358, -0.0015467, -0.0093105, -0.0011856, -0.0068110, 0.0060928
9: -0.0038763, -0.0032129, -0.0039074, -0.0032065, -0.0005257, 0.0005876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016202, upper bound: 0.0016378
time: 2.00 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016294, upper bound: 0.0016376
time: 1.83 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.0106362, -0.0037605, -0.0105667, -0.0034601, -0.0060226, 0.0052237
1: -0.0059374, -0.0039989, -0.0059178, -0.0039142, -0.0016980, 0.0014728
2: -0.0052477, 0.0090553, -0.0051031, 0.0096802, -0.0125281, 0.0108665
3: 0.0009329, 0.0028256, 0.0009520, 0.0029083, -0.0016579, 0.0014380
4: -0.0006755, 0.0100136, -0.0011425, 0.0099056, -0.0081209, 0.0093627
5: 0.9953185, 0.9982883, 0.9951888, 0.9982583, -0.0022562, 0.0026012
6: 0.0036343, 0.0063300, 0.0035165, 0.0063027, -0.0020480, 0.0023611
7: -0.0098188, 0.0002408, -0.0102584, 0.0001391, -0.0076427, 0.0088114
8: -0.0093803, -0.0015508, -0.0093011, -0.0012087, -0.0068579, 0.0059483
9: -0.0038759, -0.0032004, -0.0039055, -0.0032073, -0.0005132, 0.0005917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016318, upper bound: 0.0016386
time: 2.16 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016407, upper bound: 0.0016382
time: 1.87 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0105424, -0.0036823, -0.0105771, -0.0034117, -0.0061271, 0.0052638
1: -0.0059109, -0.0039768, -0.0059207, -0.0039006, -0.0017275, 0.0014841
2: -0.0050524, 0.0092179, -0.0051246, 0.0097808, -0.0127457, 0.0109498
3: 0.0009587, 0.0028471, 0.0009491, 0.0029216, -0.0016867, 0.0014490
4: -0.0007970, 0.0098677, -0.0012177, 0.0099217, -0.0081832, 0.0095253
5: 0.9952848, 0.9982478, 0.9951680, 0.9982628, -0.0022735, 0.0026464
6: 0.0036037, 0.0062932, 0.0034976, 0.0063068, -0.0020637, 0.0024021
7: -0.0099333, 0.0001035, -0.0103291, 0.0001543, -0.0077013, 0.0089644
8: -0.0092734, -0.0014618, -0.0093129, -0.0011537, -0.0069770, 0.0059939
9: -0.0038836, -0.0032097, -0.0039102, -0.0032063, -0.0005171, 0.0006019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016303, upper bound: 0.0016390
time: 2.11 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016388, upper bound: 0.0016390
time: 1.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.85 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 5, lower bound: -0.0016237, upper bound: 0.0016379
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 5, lower bound: -0.0016320, upper bound: 0.0016378
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 5, lower bound: -0.0016202, upper bound: 0.0016378
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 5, lower bound: -0.0016294, upper bound: 0.0016376
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 5, lower bound: -0.0016318, upper bound: 0.0016386
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 5, lower bound: -0.0016407, upper bound: 0.0016382
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 5, lower bound: -0.0016303, upper bound: 0.0016390
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 5, lower bound: -0.0016388, upper bound: 0.0016390

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0105752, -0.0038416, -0.0104615, -0.0035367, -0.0057795, 0.0052016
1: -0.0059202, -0.0040217, -0.0058881, -0.0039358, -0.0016295, 0.0014665
2: -0.0051207, 0.0088866, -0.0048841, 0.0095208, -0.0120225, 0.0108204
3: 0.0009496, 0.0028033, 0.0009810, 0.0028872, -0.0015910, 0.0014319
4: -0.0005494, 0.0099188, -0.0010234, 0.0097420, -0.0080865, 0.0089849
5: 0.9953536, 0.9982619, 0.9952219, 0.9982129, -0.0022467, 0.0024963
6: 0.0036661, 0.0063060, 0.0035466, 0.0062614, -0.0020393, 0.0022659
7: -0.0097002, 0.0001515, -0.0101462, -0.0000149, -0.0076103, 0.0084558
8: -0.0093108, -0.0016432, -0.0091813, -0.0012960, -0.0065811, 0.0059231
9: -0.0038680, -0.0032064, -0.0038979, -0.0032176, -0.0005110, 0.0005678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015762, upper bound: 0.0015876
time: 1.88 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015758, upper bound: 0.0015914
time: 1.87 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0105802, -0.0038385, -0.0104967, -0.0033800, -0.0059550, 0.0052337
1: -0.0059216, -0.0040209, -0.0058981, -0.0038916, -0.0016789, 0.0014756
2: -0.0051312, 0.0088930, -0.0049575, 0.0098467, -0.0123875, 0.0108872
3: 0.0009483, 0.0028041, 0.0009713, 0.0029304, -0.0016393, 0.0014407
4: -0.0005542, 0.0099266, -0.0012670, 0.0097968, -0.0081364, 0.0092577
5: 0.9953522, 0.9982641, 0.9951542, 0.9982280, -0.0022605, 0.0025721
6: 0.0036649, 0.0063080, 0.0034852, 0.0062753, -0.0020519, 0.0023346
7: -0.0097047, 0.0001589, -0.0103755, 0.0000367, -0.0076573, 0.0087125
8: -0.0093165, -0.0016397, -0.0092214, -0.0011176, -0.0067809, 0.0059597
9: -0.0038683, -0.0032060, -0.0039133, -0.0032142, -0.0005142, 0.0005850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015849, upper bound: 0.0015878
time: 2.08 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015849, upper bound: 0.0015915
time: 1.87 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0104812, -0.0037705, -0.0104717, -0.0034896, -0.0058995, 0.0052408
1: -0.0058937, -0.0040017, -0.0058910, -0.0039225, -0.0016633, 0.0014776
2: -0.0049253, 0.0090345, -0.0049054, 0.0096188, -0.0122721, 0.0109020
3: 0.0009755, 0.0028229, 0.0009781, 0.0029002, -0.0016240, 0.0014427
4: -0.0006600, 0.0097727, -0.0010966, 0.0097579, -0.0081475, 0.0091714
5: 0.9953228, 0.9982213, 0.9952016, 0.9982173, -0.0022636, 0.0025481
6: 0.0036382, 0.0062692, 0.0035281, 0.0062655, -0.0020547, 0.0023129
7: -0.0098042, 0.0000141, -0.0102152, 0.0000001, -0.0076677, 0.0086313
8: -0.0092038, -0.0015622, -0.0091929, -0.0012424, -0.0067178, 0.0059678
9: -0.0038750, -0.0032157, -0.0039025, -0.0032166, -0.0005149, 0.0005796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015723, upper bound: 0.0015948
time: 1.88 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015776, upper bound: 0.0015943
time: 1.87 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0104850, -0.0037673, -0.0105069, -0.0033337, -0.0060720, 0.0052749
1: -0.0058948, -0.0040008, -0.0059010, -0.0038786, -0.0017119, 0.0014872
2: -0.0049330, 0.0090412, -0.0049787, 0.0099431, -0.0126310, 0.0109729
3: 0.0009745, 0.0028237, 0.0009684, 0.0029431, -0.0016715, 0.0014521
4: -0.0006649, 0.0097785, -0.0013390, 0.0098127, -0.0082005, 0.0094396
5: 0.9953215, 0.9982231, 0.9951342, 0.9982325, -0.0022783, 0.0026226
6: 0.0036370, 0.0062707, 0.0034670, 0.0062793, -0.0020680, 0.0023805
7: -0.0098089, 0.0000195, -0.0104432, 0.0000517, -0.0077176, 0.0088838
8: -0.0092081, -0.0015586, -0.0092331, -0.0010649, -0.0069142, 0.0060066
9: -0.0038753, -0.0032153, -0.0039179, -0.0032132, -0.0005182, 0.0005965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015822, upper bound: 0.0015946
time: 1.41 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015868, upper bound: 0.0015943
time: 1.99 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0106073, -0.0037736, -0.0104636, -0.0035097, -0.0059398, 0.0051177
1: -0.0059293, -0.0040026, -0.0058887, -0.0039282, -0.0016747, 0.0014429
2: -0.0051875, 0.0090281, -0.0048886, 0.0095770, -0.0123560, 0.0106459
3: 0.0009408, 0.0028220, 0.0009804, 0.0028947, -0.0016351, 0.0014088
4: -0.0006552, 0.0099687, -0.0010654, 0.0097453, -0.0079561, 0.0092341
5: 0.9953242, 0.9982758, 0.9952102, 0.9982138, -0.0022104, 0.0025655
6: 0.0036394, 0.0063186, 0.0035360, 0.0062623, -0.0020064, 0.0023287
7: -0.0097997, 0.0001985, -0.0101858, -0.0000117, -0.0074876, 0.0086903
8: -0.0093473, -0.0015657, -0.0091838, -0.0012652, -0.0067637, 0.0058276
9: -0.0038747, -0.0032033, -0.0039006, -0.0032174, -0.0005028, 0.0005835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015823, upper bound: 0.0015881
time: 1.48 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015823, upper bound: 0.0015916
time: 1.87 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0106130, -0.0037705, -0.0104988, -0.0033503, -0.0061143, 0.0051516
1: -0.0059309, -0.0040017, -0.0058987, -0.0038832, -0.0017239, 0.0014524
2: -0.0051993, 0.0090344, -0.0049617, 0.0099086, -0.0127191, 0.0107165
3: 0.0009393, 0.0028229, 0.0009707, 0.0029385, -0.0016832, 0.0014182
4: -0.0006599, 0.0099775, -0.0013132, 0.0097999, -0.0080088, 0.0095054
5: 0.9953229, 0.9982783, 0.9951414, 0.9982290, -0.0022251, 0.0026409
6: 0.0036383, 0.0063208, 0.0034735, 0.0062761, -0.0020197, 0.0023971
7: -0.0098042, 0.0002068, -0.0104190, 0.0000397, -0.0075372, 0.0089457
8: -0.0093538, -0.0015623, -0.0092238, -0.0010837, -0.0069624, 0.0058662
9: -0.0038750, -0.0032027, -0.0039162, -0.0032140, -0.0005061, 0.0006007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015880
time: 1.71 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015919
time: 1.76 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0105141, -0.0036957, -0.0104738, -0.0034617, -0.0060483, 0.0051582
1: -0.0059030, -0.0039806, -0.0058916, -0.0039146, -0.0017052, 0.0014543
2: -0.0049936, 0.0091901, -0.0049098, 0.0096769, -0.0125816, 0.0107301
3: 0.0009665, 0.0028435, 0.0009776, 0.0029079, -0.0016650, 0.0014200
4: -0.0007763, 0.0098238, -0.0011400, 0.0097612, -0.0080190, 0.0094027
5: 0.9952906, 0.9982355, 0.9951895, 0.9982181, -0.0022279, 0.0026124
6: 0.0036089, 0.0062821, 0.0035172, 0.0062663, -0.0020223, 0.0023712
7: -0.0099137, 0.0000621, -0.0102560, 0.0000032, -0.0075468, 0.0088490
8: -0.0092412, -0.0014770, -0.0091954, -0.0012106, -0.0068872, 0.0058737
9: -0.0038823, -0.0032124, -0.0039053, -0.0032164, -0.0005068, 0.0005942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015861, upper bound: 0.0015917
time: 1.36 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015861, upper bound: 0.0015946
time: 1.88 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0105185, -0.0036925, -0.0105089, -0.0033041, -0.0062140, 0.0051906
1: -0.0059042, -0.0039797, -0.0059015, -0.0038702, -0.0017520, 0.0014634
2: -0.0050028, 0.0091967, -0.0049829, 0.0100046, -0.0129264, 0.0107976
3: 0.0009652, 0.0028443, 0.0009679, 0.0029513, -0.0017106, 0.0014289
4: -0.0007812, 0.0098307, -0.0013850, 0.0098157, -0.0080694, 0.0096604
5: 0.9952892, 0.9982375, 0.9951214, 0.9982334, -0.0022419, 0.0026839
6: 0.0036077, 0.0062838, 0.0034554, 0.0062801, -0.0020350, 0.0024362
7: -0.0099183, 0.0000686, -0.0104866, 0.0000546, -0.0075942, 0.0090915
8: -0.0092463, -0.0014734, -0.0092353, -0.0010312, -0.0070759, 0.0059106
9: -0.0038826, -0.0032120, -0.0039208, -0.0032130, -0.0005099, 0.0006105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015914
time: 1.90 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015951
time: 1.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.98 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015762, upper bound: 0.0015876
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015758, upper bound: 0.0015914
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015849, upper bound: 0.0015878
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015849, upper bound: 0.0015915
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015723, upper bound: 0.0015948
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015776, upper bound: 0.0015943
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015822, upper bound: 0.0015946
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015868, upper bound: 0.0015943
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015823, upper bound: 0.0015881
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015823, upper bound: 0.0015916
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015880
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015919
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015861, upper bound: 0.0015917
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015861, upper bound: 0.0015946
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015914
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015951

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0105717, -0.0038555, -0.0104371, -0.0036374, -0.0056615, 0.0051614
1: -0.0059192, -0.0040257, -0.0058813, -0.0039642, -0.0015962, 0.0014552
2: -0.0051135, 0.0088577, -0.0048335, 0.0093113, -0.0117771, 0.0107367
3: 0.0009506, 0.0027995, 0.0009877, 0.0028595, -0.0015585, 0.0014208
4: -0.0005278, 0.0099134, -0.0008668, 0.0097041, -0.0080240, 0.0088015
5: 0.9953596, 0.9982605, 0.9952654, 0.9982023, -0.0022293, 0.0024453
6: 0.0036716, 0.0063047, 0.0035861, 0.0062519, -0.0020235, 0.0022196
7: -0.0096799, 0.0001465, -0.0099989, -0.0000505, -0.0075514, 0.0082832
8: -0.0093069, -0.0016590, -0.0091536, -0.0014107, -0.0064468, 0.0058773
9: -0.0038666, -0.0032068, -0.0038880, -0.0032200, -0.0005071, 0.0005562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015430, upper bound: 0.0015564
time: 1.84 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015415, upper bound: 0.0015566
time: 1.83 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0105619, -0.0038994, -0.0106190, -0.0036188, -0.0057024, 0.0053859
1: -0.0059164, -0.0040381, -0.0059326, -0.0039589, -0.0016077, 0.0015185
2: -0.0050930, 0.0087662, -0.0052119, 0.0093500, -0.0118621, 0.0112037
3: 0.0009533, 0.0027874, 0.0009376, 0.0028646, -0.0015698, 0.0014826
4: -0.0004594, 0.0098981, -0.0008958, 0.0099869, -0.0083729, 0.0088650
5: 0.9953786, 0.9982562, 0.9952574, 0.9982809, -0.0023263, 0.0024630
6: 0.0036888, 0.0063008, 0.0035788, 0.0063232, -0.0021115, 0.0022356
7: -0.0096155, 0.0001320, -0.0100261, 0.0002157, -0.0078799, 0.0083429
8: -0.0092956, -0.0017091, -0.0093607, -0.0013895, -0.0064933, 0.0061329
9: -0.0038623, -0.0032078, -0.0038899, -0.0032021, -0.0005291, 0.0005602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015430, upper bound: 0.0015597
time: 2.02 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015415, upper bound: 0.0015594
time: 1.94 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0105768, -0.0038524, -0.0104722, -0.0034750, -0.0058425, 0.0051931
1: -0.0059206, -0.0040248, -0.0058912, -0.0039184, -0.0016472, 0.0014641
2: -0.0051240, 0.0088641, -0.0049064, 0.0096492, -0.0121536, 0.0108027
3: 0.0009492, 0.0028003, 0.0009780, 0.0029042, -0.0016083, 0.0014296
4: -0.0005326, 0.0099212, -0.0011193, 0.0097586, -0.0080733, 0.0090829
5: 0.9953583, 0.9982627, 0.9951952, 0.9982175, -0.0022430, 0.0025235
6: 0.0036703, 0.0063067, 0.0035224, 0.0062656, -0.0020360, 0.0022906
7: -0.0096844, 0.0001538, -0.0102365, 0.0000008, -0.0075979, 0.0085480
8: -0.0093126, -0.0016555, -0.0091935, -0.0012257, -0.0066529, 0.0059134
9: -0.0038669, -0.0032063, -0.0039040, -0.0032166, -0.0005102, 0.0005740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015500, upper bound: 0.0015566
time: 1.86 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015525, upper bound: 0.0015566
time: 1.51 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0105669, -0.0038964, -0.0106496, -0.0034668, -0.0058807, 0.0054178
1: -0.0059179, -0.0040372, -0.0059412, -0.0039161, -0.0016580, 0.0015275
2: -0.0051035, 0.0087726, -0.0052754, 0.0096663, -0.0122330, 0.0112701
3: 0.0009519, 0.0027882, 0.0009292, 0.0029065, -0.0016188, 0.0014914
4: -0.0004643, 0.0099059, -0.0011321, 0.0100344, -0.0084226, 0.0091422
5: 0.9953772, 0.9982584, 0.9951918, 0.9982941, -0.0023400, 0.0025400
6: 0.0036876, 0.0063028, 0.0035192, 0.0063352, -0.0021241, 0.0023055
7: -0.0096201, 0.0001394, -0.0102486, 0.0002603, -0.0079266, 0.0086038
8: -0.0093014, -0.0017056, -0.0093955, -0.0012164, -0.0066963, 0.0061693
9: -0.0038626, -0.0032073, -0.0039048, -0.0031991, -0.0005323, 0.0005777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015499, upper bound: 0.0015597
time: 2.08 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015525, upper bound: 0.0015597
time: 1.86 seconds

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0104564, -0.0038748, -0.0104683, -0.0035039, -0.0058566, 0.0051174
1: -0.0058867, -0.0040311, -0.0058901, -0.0039266, -0.0016512, 0.0014428
2: -0.0048735, 0.0088175, -0.0048984, 0.0095889, -0.0121829, 0.0106453
3: 0.0009824, 0.0027941, 0.0009791, 0.0028962, -0.0016122, 0.0014087
4: -0.0004977, 0.0097340, -0.0010743, 0.0097526, -0.0079557, 0.0091047
5: 0.9953679, 0.9982107, 0.9952078, 0.9982158, -0.0022103, 0.0025296
6: 0.0036791, 0.0062595, 0.0035337, 0.0062641, -0.0020063, 0.0022961
7: -0.0096516, -0.0000223, -0.0101942, -0.0000048, -0.0074872, 0.0085686
8: -0.0091755, -0.0016810, -0.0091891, -0.0012587, -0.0066689, 0.0058273
9: -0.0038647, -0.0032181, -0.0039011, -0.0032169, -0.0005027, 0.0005754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015396, upper bound: 0.0015639
time: 1.88 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015402, upper bound: 0.0015634
time: 1.86 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0106379, -0.0038413, -0.0104590, -0.0035472, -0.0060869, 0.0051866
1: -0.0059379, -0.0040217, -0.0058874, -0.0039387, -0.0017161, 0.0014623
2: -0.0052511, 0.0088872, -0.0048790, 0.0094990, -0.0126619, 0.0107893
3: 0.0009324, 0.0028034, 0.0009816, 0.0028843, -0.0016756, 0.0014278
4: -0.0005499, 0.0100162, -0.0010071, 0.0097381, -0.0080632, 0.0094627
5: 0.9953534, 0.9982890, 0.9952264, 0.9982117, -0.0022402, 0.0026290
6: 0.0036660, 0.0063306, 0.0035507, 0.0062605, -0.0020334, 0.0023864
7: -0.0097006, 0.0002432, -0.0101309, -0.0000185, -0.0075884, 0.0089055
8: -0.0093822, -0.0016429, -0.0091785, -0.0013080, -0.0069312, 0.0059061
9: -0.0038680, -0.0032003, -0.0038969, -0.0032179, -0.0005095, 0.0005980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015453, upper bound: 0.0015609
time: 1.95 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015453, upper bound: 0.0015637
time: 2.06 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0104601, -0.0038717, -0.0105036, -0.0033473, -0.0060299, 0.0051516
1: -0.0058878, -0.0040302, -0.0059000, -0.0038824, -0.0017001, 0.0014524
2: -0.0048813, 0.0088239, -0.0049717, 0.0099148, -0.0125435, 0.0107164
3: 0.0009813, 0.0027950, 0.0009694, 0.0029394, -0.0016599, 0.0014181
4: -0.0005026, 0.0097399, -0.0013178, 0.0098074, -0.0080088, 0.0093742
5: 0.9953666, 0.9982122, 0.9951401, 0.9982311, -0.0022251, 0.0026044
6: 0.0036779, 0.0062609, 0.0034723, 0.0062780, -0.0020197, 0.0023640
7: -0.0096561, -0.0000168, -0.0104234, 0.0000467, -0.0075372, 0.0088222
8: -0.0091798, -0.0016775, -0.0092292, -0.0010803, -0.0068663, 0.0058662
9: -0.0038650, -0.0032178, -0.0039165, -0.0032135, -0.0005061, 0.0005924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015464, upper bound: 0.0015634
time: 1.91 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015521, upper bound: 0.0015637
time: 1.94 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0106428, -0.0038382, -0.0104941, -0.0033939, -0.0062517, 0.0052203
1: -0.0059393, -0.0040208, -0.0058973, -0.0038955, -0.0017626, 0.0014718
2: -0.0052614, 0.0088936, -0.0049520, 0.0098178, -0.0130048, 0.0108593
3: 0.0009310, 0.0028042, 0.0009720, 0.0029265, -0.0017210, 0.0014371
4: -0.0005546, 0.0100239, -0.0012454, 0.0097927, -0.0081156, 0.0097189
5: 0.9953522, 0.9982913, 0.9951602, 0.9982269, -0.0022548, 0.0027002
6: 0.0036648, 0.0063325, 0.0034906, 0.0062742, -0.0020466, 0.0024510
7: -0.0097051, 0.0002505, -0.0103552, 0.0000329, -0.0076377, 0.0091466
8: -0.0093878, -0.0016394, -0.0092184, -0.0011334, -0.0071188, 0.0059444
9: -0.0038683, -0.0031998, -0.0039119, -0.0032144, -0.0005129, 0.0006142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015539, upper bound: 0.0015639
time: 1.94 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015636
time: 1.48 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0106039, -0.0037872, -0.0104394, -0.0036104, -0.0058243, 0.0050771
1: -0.0059283, -0.0040064, -0.0058819, -0.0039566, -0.0016421, 0.0014314
2: -0.0051804, 0.0089997, -0.0048383, 0.0093676, -0.0121158, 0.0105614
3: 0.0009417, 0.0028183, 0.0009870, 0.0028669, -0.0016033, 0.0013976
4: -0.0006340, 0.0099634, -0.0009089, 0.0097077, -0.0078929, 0.0090546
5: 0.9953301, 0.9982743, 0.9952537, 0.9982034, -0.0021929, 0.0025156
6: 0.0036448, 0.0063173, 0.0035755, 0.0062528, -0.0019905, 0.0022834
7: -0.0097798, 0.0001935, -0.0100385, -0.0000471, -0.0074281, 0.0085214
8: -0.0093435, -0.0015813, -0.0091562, -0.0013799, -0.0066322, 0.0057813
9: -0.0038733, -0.0032036, -0.0038907, -0.0032198, -0.0004988, 0.0005722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015484, upper bound: 0.0015564
time: 1.77 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015484, upper bound: 0.0015564
time: 1.97 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0105941, -0.0038326, -0.0106212, -0.0035921, -0.0058632, 0.0053276
1: -0.0059255, -0.0040192, -0.0059332, -0.0039514, -0.0016530, 0.0015020
2: -0.0051600, 0.0089052, -0.0052164, 0.0094056, -0.0121966, 0.0110824
3: 0.0009445, 0.0028058, 0.0009370, 0.0028720, -0.0016140, 0.0014666
4: -0.0005634, 0.0099481, -0.0009373, 0.0099903, -0.0082823, 0.0091150
5: 0.9953497, 0.9982701, 0.9952459, 0.9982818, -0.0023011, 0.0025324
6: 0.0036626, 0.0063134, 0.0035683, 0.0063241, -0.0020887, 0.0022987
7: -0.0097133, 0.0001791, -0.0100652, 0.0002188, -0.0077946, 0.0085782
8: -0.0093323, -0.0016330, -0.0093632, -0.0013591, -0.0066764, 0.0060665
9: -0.0038689, -0.0032046, -0.0038925, -0.0032019, -0.0005234, 0.0005760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015485, upper bound: 0.0015595
time: 1.87 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015484, upper bound: 0.0015598
time: 1.60 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0106096, -0.0037842, -0.0104743, -0.0034461, -0.0060021, 0.0051106
1: -0.0059299, -0.0040056, -0.0058917, -0.0039102, -0.0016922, 0.0014409
2: -0.0051922, 0.0090060, -0.0049107, 0.0097093, -0.0124857, 0.0106310
3: 0.0009402, 0.0028191, 0.0009774, 0.0029122, -0.0016523, 0.0014068
4: -0.0006387, 0.0099722, -0.0011643, 0.0097618, -0.0079450, 0.0093310
5: 0.9953288, 0.9982769, 0.9951828, 0.9982184, -0.0022073, 0.0025924
6: 0.0036436, 0.0063195, 0.0035111, 0.0062665, -0.0020036, 0.0023531
7: -0.0097842, 0.0002018, -0.0102789, 0.0000038, -0.0074771, 0.0087815
8: -0.0093499, -0.0015778, -0.0091959, -0.0011928, -0.0068347, 0.0058194
9: -0.0038736, -0.0032031, -0.0039068, -0.0032164, -0.0005021, 0.0005897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015555, upper bound: 0.0015567
time: 1.73 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015566
time: 1.36 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0105998, -0.0038296, -0.0106515, -0.0034403, -0.0060396, 0.0053604
1: -0.0059271, -0.0040184, -0.0059417, -0.0039086, -0.0017028, 0.0015113
2: -0.0051719, 0.0089115, -0.0052795, 0.0097214, -0.0125635, 0.0111507
3: 0.0009429, 0.0028066, 0.0009286, 0.0029138, -0.0016626, 0.0014756
4: -0.0005681, 0.0099570, -0.0011733, 0.0100374, -0.0083334, 0.0093892
5: 0.9953484, 0.9982726, 0.9951803, 0.9982949, -0.0023153, 0.0026086
6: 0.0036614, 0.0063157, 0.0035088, 0.0063360, -0.0021016, 0.0023678
7: -0.0097177, 0.0001875, -0.0102873, 0.0002632, -0.0078426, 0.0088363
8: -0.0093388, -0.0016295, -0.0093977, -0.0011862, -0.0068773, 0.0061039
9: -0.0038691, -0.0032040, -0.0039074, -0.0031989, -0.0005266, 0.0005933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015555, upper bound: 0.0015598
time: 1.87 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015598
time: 1.85 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0105107, -0.0037102, -0.0104494, -0.0035633, -0.0059288, 0.0051159
1: -0.0059020, -0.0039847, -0.0058847, -0.0039433, -0.0016716, 0.0014424
2: -0.0049866, 0.0091599, -0.0048590, 0.0094655, -0.0123331, 0.0106421
3: 0.0009674, 0.0028395, 0.0009843, 0.0028799, -0.0016321, 0.0014083
4: -0.0007537, 0.0098185, -0.0009820, 0.0097232, -0.0079533, 0.0092170
5: 0.9952968, 0.9982342, 0.9952334, 0.9982076, -0.0022097, 0.0025608
6: 0.0036146, 0.0062808, 0.0035570, 0.0062567, -0.0020057, 0.0023244
7: -0.0098924, 0.0000572, -0.0101073, -0.0000326, -0.0074849, 0.0086742
8: -0.0092374, -0.0014936, -0.0091675, -0.0013263, -0.0067512, 0.0058255
9: -0.0038809, -0.0032128, -0.0038953, -0.0032188, -0.0005026, 0.0005825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015537, upper bound: 0.0015611
time: 1.34 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015531, upper bound: 0.0015608
time: 1.86 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0105013, -0.0037521, -0.0106312, -0.0035424, -0.0059686, 0.0053561
1: -0.0058994, -0.0039965, -0.0059360, -0.0039374, -0.0016828, 0.0015101
2: -0.0049670, 0.0090727, -0.0052372, 0.0095090, -0.0124159, 0.0111418
3: 0.0009700, 0.0028279, 0.0009342, 0.0028857, -0.0016430, 0.0014744
4: -0.0006885, 0.0098039, -0.0010146, 0.0100058, -0.0083267, 0.0092788
5: 0.9953150, 0.9982300, 0.9952244, 0.9982861, -0.0023134, 0.0025779
6: 0.0036310, 0.0062771, 0.0035488, 0.0063280, -0.0020999, 0.0023400
7: -0.0098311, 0.0000434, -0.0101379, 0.0002334, -0.0078364, 0.0087324
8: -0.0092267, -0.0015413, -0.0093745, -0.0013025, -0.0067965, 0.0060990
9: -0.0038768, -0.0032137, -0.0038974, -0.0032009, -0.0005262, 0.0005864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015537, upper bound: 0.0015639
time: 1.95 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015532, upper bound: 0.0015639
time: 2.26 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0105152, -0.0037070, -0.0104843, -0.0034013, -0.0060977, 0.0051483
1: -0.0059033, -0.0039838, -0.0058946, -0.0038976, -0.0017192, 0.0014515
2: -0.0049959, 0.0091664, -0.0049315, 0.0098025, -0.0126844, 0.0107096
3: 0.0009662, 0.0028403, 0.0009747, 0.0029245, -0.0016786, 0.0014172
4: -0.0007586, 0.0098255, -0.0012339, 0.0097774, -0.0080037, 0.0094795
5: 0.9952955, 0.9982361, 0.9951635, 0.9982226, -0.0022237, 0.0026337
6: 0.0036134, 0.0062825, 0.0034935, 0.0062704, -0.0020184, 0.0023906
7: -0.0098970, 0.0000637, -0.0103444, 0.0000185, -0.0075323, 0.0089213
8: -0.0092425, -0.0014900, -0.0092072, -0.0011418, -0.0069434, 0.0058624
9: -0.0038812, -0.0032123, -0.0039112, -0.0032154, -0.0005058, 0.0005990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015607, upper bound: 0.0015605
time: 1.76 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015608
time: 1.88 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0105058, -0.0037490, -0.0106614, -0.0033909, -0.0061359, 0.0053885
1: -0.0059006, -0.0039956, -0.0059445, -0.0038947, -0.0017299, 0.0015192
2: -0.0049764, 0.0090792, -0.0053001, 0.0098240, -0.0127640, 0.0112093
3: 0.0009688, 0.0028288, 0.0009259, 0.0029274, -0.0016891, 0.0014834
4: -0.0006934, 0.0098109, -0.0012500, 0.0100528, -0.0083771, 0.0095390
5: 0.9953136, 0.9982321, 0.9951590, 0.9982992, -0.0023274, 0.0026502
6: 0.0036298, 0.0062788, 0.0034894, 0.0063398, -0.0021126, 0.0024056
7: -0.0098357, 0.0000500, -0.0103595, 0.0002777, -0.0078838, 0.0089773
8: -0.0092318, -0.0015377, -0.0094090, -0.0011300, -0.0069870, 0.0061360
9: -0.0038771, -0.0032133, -0.0039122, -0.0031980, -0.0005294, 0.0006028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015607, upper bound: 0.0015637
time: 1.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015634
time: 1.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.59 seconds
IS_A1_A1_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015430, upper bound: 0.0015564
IS_A1_A1_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015415, upper bound: 0.0015566
IS_A1_A1_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015430, upper bound: 0.0015597
IS_A1_A1_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015415, upper bound: 0.0015594
IS_A1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015500, upper bound: 0.0015566
IS_A1_A1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015525, upper bound: 0.0015566
IS_A1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015499, upper bound: 0.0015597
IS_A1_A1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015525, upper bound: 0.0015597
IS_A1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015396, upper bound: 0.0015639
IS_A1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015402, upper bound: 0.0015634
IS_A1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015453, upper bound: 0.0015609
IS_A1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015453, upper bound: 0.0015637
IS_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015464, upper bound: 0.0015634
IS_A1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015521, upper bound: 0.0015637
IS_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015539, upper bound: 0.0015639
IS_A1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015636
IS_A2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015484, upper bound: 0.0015564
IS_A2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015484, upper bound: 0.0015564
IS_A2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015485, upper bound: 0.0015595
IS_A2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015484, upper bound: 0.0015598
IS_A2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015555, upper bound: 0.0015567
IS_A2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015566
IS_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015555, upper bound: 0.0015598
IS_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015598
IS_A2_A2_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015537, upper bound: 0.0015611
IS_A2_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015531, upper bound: 0.0015608
IS_A2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015537, upper bound: 0.0015639
IS_A2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015532, upper bound: 0.0015639
IS_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015607, upper bound: 0.0015605
IS_A2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015608
IS_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015607, upper bound: 0.0015637
IS_A2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.59
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015634

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 4.26 + 153.15 = 157.41 seconds
