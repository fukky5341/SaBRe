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
execution time: IAR + RelationalAnalysis = 1.36 + 2.93 = 4.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0017475, upper bound: 0.0017472

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016955, upper bound: 0.0017030
time: 1.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0017045, upper bound: 0.0017047
time: 1.70 seconds

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
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016503, upper bound: 0.0016608
time: 1.91 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016505, upper bound: 0.0016585
time: 1.88 seconds

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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016598, upper bound: 0.0016620
time: 1.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016601, upper bound: 0.0016602
time: 1.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.94 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.94
Output dim: 5, lower bound: -0.0016503, upper bound: 0.0016608
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.94
Output dim: 5, lower bound: -0.0016505, upper bound: 0.0016585
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.94
Output dim: 5, lower bound: -0.0016598, upper bound: 0.0016620
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.94
Output dim: 5, lower bound: -0.0016601, upper bound: 0.0016602

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0105175, -0.0036731, -0.0106510, -0.0036513, -0.0056822, 0.0054853
1: -0.0059039, -0.0039742, -0.0059416, -0.0039681, -0.0016020, 0.0015465
2: -0.0050006, 0.0092371, -0.0052783, 0.0092824, -0.0118200, 0.0114105
3: 0.0009655, 0.0028497, 0.0009288, 0.0028557, -0.0015642, 0.0015100
4: -0.0008113, 0.0098290, -0.0008452, 0.0100366, -0.0085275, 0.0088336
5: 0.9952808, 0.9982370, 0.9952714, 0.9982947, -0.0023692, 0.0024542
6: 0.0036001, 0.0062834, 0.0035915, 0.0063357, -0.0021505, 0.0022277
7: -0.0099467, 0.0000671, -0.0099786, 0.0002624, -0.0080253, 0.0083134
8: -0.0092451, -0.0014513, -0.0093971, -0.0014265, -0.0064703, 0.0062461
9: -0.0038845, -0.0032121, -0.0038867, -0.0031990, -0.0005389, 0.0005582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015974, upper bound: 0.0016103
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016022, upper bound: 0.0016103
time: 1.44 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0105284, -0.0036278, -0.0105565, -0.0035738, -0.0057154, 0.0055854
1: -0.0059070, -0.0039615, -0.0059149, -0.0039462, -0.0016114, 0.0015747
2: -0.0050233, 0.0093312, -0.0050819, 0.0094436, -0.0118892, 0.0116188
3: 0.0009625, 0.0028621, 0.0009548, 0.0028770, -0.0015733, 0.0015376
4: -0.0008817, 0.0098459, -0.0009657, 0.0098897, -0.0086831, 0.0088852
5: 0.9952612, 0.9982417, 0.9952379, 0.9982539, -0.0024124, 0.0024686
6: 0.0035823, 0.0062877, 0.0035611, 0.0062987, -0.0021898, 0.0022407
7: -0.0100129, 0.0000830, -0.0100920, 0.0001242, -0.0081718, 0.0083620
8: -0.0092575, -0.0013998, -0.0092895, -0.0013383, -0.0065082, 0.0063601
9: -0.0038890, -0.0032110, -0.0038943, -0.0032083, -0.0005487, 0.0005615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016499, upper bound: 0.0016582
time: 1.98 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016499, upper bound: 0.0016584
time: 2.14 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0105505, -0.0036010, -0.0106533, -0.0036244, -0.0058317, 0.0057257
1: -0.0059133, -0.0039539, -0.0059422, -0.0039605, -0.0016442, 0.0016143
2: -0.0050694, 0.0093870, -0.0052831, 0.0093383, -0.0121311, 0.0119107
3: 0.0009564, 0.0028695, 0.0009282, 0.0028631, -0.0016054, 0.0015762
4: -0.0009234, 0.0098804, -0.0008870, 0.0100401, -0.0089013, 0.0090661
5: 0.9952497, 0.9982513, 0.9952597, 0.9982956, -0.0024731, 0.0025188
6: 0.0035718, 0.0062964, 0.0035810, 0.0063366, -0.0022448, 0.0022863
7: -0.0100521, 0.0001155, -0.0100179, 0.0002657, -0.0083771, 0.0085322
8: -0.0092827, -0.0013693, -0.0093997, -0.0013959, -0.0066406, 0.0065199
9: -0.0038916, -0.0032089, -0.0038893, -0.0031988, -0.0005625, 0.0005729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016385, upper bound: 0.0016315
time: 2.15 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016385, upper bound: 0.0016409
time: 1.88 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0105609, -0.0035510, -0.0105587, -0.0035466, -0.0058762, 0.0058426
1: -0.0059162, -0.0039398, -0.0059156, -0.0039386, -0.0016567, 0.0016473
2: -0.0050910, 0.0094910, -0.0050864, 0.0095001, -0.0122238, 0.0121539
3: 0.0009536, 0.0028833, 0.0009542, 0.0028845, -0.0016176, 0.0016084
4: -0.0010011, 0.0098966, -0.0010079, 0.0098931, -0.0090831, 0.0091353
5: 0.9952281, 0.9982558, 0.9952263, 0.9982548, -0.0025235, 0.0025380
6: 0.0035522, 0.0063004, 0.0035505, 0.0062996, -0.0022906, 0.0023038
7: -0.0101253, 0.0001306, -0.0101317, 0.0001274, -0.0085482, 0.0085973
8: -0.0092945, -0.0013123, -0.0092920, -0.0013073, -0.0066913, 0.0066531
9: -0.0038965, -0.0032078, -0.0038969, -0.0032081, -0.0005740, 0.0005773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016589, upper bound: 0.0016595
time: 1.81 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016589, upper bound: 0.0016604
time: 2.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.33 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 5, lower bound: -0.0015974, upper bound: 0.0016103
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 5, lower bound: -0.0016022, upper bound: 0.0016103
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 5, lower bound: -0.0016499, upper bound: 0.0016582
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 5, lower bound: -0.0016499, upper bound: 0.0016584
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 5, lower bound: -0.0016385, upper bound: 0.0016315
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 5, lower bound: -0.0016385, upper bound: 0.0016409
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 5, lower bound: -0.0016589, upper bound: 0.0016595
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 5, lower bound: -0.0016589, upper bound: 0.0016604

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0104926, -0.0037756, -0.0106476, -0.0036650, -0.0056404, 0.0053625
1: -0.0058969, -0.0040031, -0.0059406, -0.0039720, -0.0015902, 0.0015119
2: -0.0049489, 0.0090238, -0.0052714, 0.0092539, -0.0117332, 0.0111551
3: 0.0009724, 0.0028215, 0.0009297, 0.0028519, -0.0015527, 0.0014762
4: -0.0006520, 0.0097903, -0.0008240, 0.0100313, -0.0083367, 0.0087687
5: 0.9953251, 0.9982263, 0.9952773, 0.9982933, -0.0023162, 0.0024362
6: 0.0036402, 0.0062736, 0.0035969, 0.0063344, -0.0021024, 0.0022113
7: -0.0097967, 0.0000307, -0.0099586, 0.0002575, -0.0078457, 0.0082523
8: -0.0092167, -0.0015681, -0.0093933, -0.0014421, -0.0064228, 0.0061063
9: -0.0038745, -0.0032146, -0.0038853, -0.0031993, -0.0005268, 0.0005541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015974, upper bound: 0.0016027
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015974, upper bound: 0.0016103
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0106743, -0.0037516, -0.0106381, -0.0037101, -0.0058656, 0.0054280
1: -0.0059481, -0.0039964, -0.0059379, -0.0039847, -0.0016537, 0.0015304
2: -0.0053268, 0.0090737, -0.0052515, 0.0091601, -0.0122016, 0.0112914
3: 0.0009224, 0.0028281, 0.0009323, 0.0028395, -0.0016147, 0.0014942
4: -0.0006892, 0.0100728, -0.0007539, 0.0100165, -0.0084385, 0.0091187
5: 0.9953148, 0.9983047, 0.9952967, 0.9982892, -0.0023445, 0.0025334
6: 0.0036309, 0.0063449, 0.0036146, 0.0063307, -0.0021281, 0.0022996
7: -0.0098318, 0.0002965, -0.0098926, 0.0002435, -0.0079416, 0.0085817
8: -0.0094236, -0.0015408, -0.0093824, -0.0014934, -0.0066792, 0.0061809
9: -0.0038768, -0.0031967, -0.0038809, -0.0032003, -0.0005333, 0.0005762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016022, upper bound: 0.0016027
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016022, upper bound: 0.0016099
time: 1.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0106038, -0.0038284, -0.0105565, -0.0035738, -0.0058722, 0.0052963
1: -0.0059283, -0.0040180, -0.0059149, -0.0039462, -0.0016556, 0.0014932
2: -0.0051802, 0.0089141, -0.0050819, 0.0094436, -0.0122154, 0.0110173
3: 0.0009418, 0.0028069, 0.0009548, 0.0028770, -0.0016165, 0.0014580
4: -0.0005700, 0.0099632, -0.0009657, 0.0098897, -0.0082336, 0.0091290
5: 0.9953479, 0.9982744, 0.9952379, 0.9982539, -0.0022875, 0.0025363
6: 0.0036609, 0.0063173, 0.0035611, 0.0062987, -0.0020764, 0.0023022
7: -0.0097195, 0.0001934, -0.0100920, 0.0001242, -0.0077488, 0.0085914
8: -0.0093434, -0.0016281, -0.0092895, -0.0013383, -0.0066867, 0.0060309
9: -0.0038693, -0.0032036, -0.0038943, -0.0032083, -0.0005203, 0.0005769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016495, upper bound: 0.0016505
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016495, upper bound: 0.0016585
time: 1.92 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0105093, -0.0037568, -0.0105565, -0.0035738, -0.0056983, 0.0053340
1: -0.0059016, -0.0039978, -0.0059149, -0.0039462, -0.0016066, 0.0015038
2: -0.0049837, 0.0090629, -0.0050819, 0.0094436, -0.0118536, 0.0110958
3: 0.0009678, 0.0028266, 0.0009548, 0.0028770, -0.0015686, 0.0014683
4: -0.0006812, 0.0098164, -0.0009657, 0.0098897, -0.0082923, 0.0088587
5: 0.9953170, 0.9982336, 0.9952379, 0.9982539, -0.0023038, 0.0024612
6: 0.0036329, 0.0062802, 0.0035611, 0.0062987, -0.0020912, 0.0022340
7: -0.0098242, 0.0000552, -0.0100920, 0.0001242, -0.0078040, 0.0083370
8: -0.0092358, -0.0015467, -0.0092895, -0.0013383, -0.0064887, 0.0060738
9: -0.0038763, -0.0032129, -0.0038943, -0.0032083, -0.0005240, 0.0005598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016495, upper bound: 0.0016499
time: 2.22 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016495, upper bound: 0.0016578
time: 2.04 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0104474, -0.0036504, -0.0106244, -0.0036375, -0.0057036, 0.0056439
1: -0.0058842, -0.0039678, -0.0059341, -0.0039642, -0.0016081, 0.0015912
2: -0.0048550, 0.0092843, -0.0052230, 0.0093110, -0.0118647, 0.0117406
3: 0.0009848, 0.0028559, 0.0009361, 0.0028595, -0.0015701, 0.0015537
4: -0.0008466, 0.0097202, -0.0008666, 0.0099952, -0.0087742, 0.0088670
5: 0.9952711, 0.9982068, 0.9952654, 0.9982832, -0.0024377, 0.0024635
6: 0.0035912, 0.0062560, 0.0035861, 0.0063253, -0.0022127, 0.0022361
7: -0.0099799, -0.0000354, -0.0099987, 0.0002235, -0.0082575, 0.0083448
8: -0.0091653, -0.0014255, -0.0093668, -0.0014108, -0.0064948, 0.0064268
9: -0.0038868, -0.0032190, -0.0038880, -0.0032016, -0.0005545, 0.0005603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016376, upper bound: 0.0016236
time: 2.16 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016376, upper bound: 0.0016257
time: 2.28 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0104822, -0.0034909, -0.0106301, -0.0036345, -0.0057489, 0.0058239
1: -0.0058940, -0.0039229, -0.0059357, -0.0039634, -0.0016208, 0.0016420
2: -0.0049272, 0.0096160, -0.0052349, 0.0093173, -0.0119589, 0.0121149
3: 0.0009753, 0.0028998, 0.0009345, 0.0028603, -0.0015826, 0.0016032
4: -0.0010946, 0.0097742, -0.0008713, 0.0100041, -0.0090539, 0.0089374
5: 0.9952021, 0.9982219, 0.9952642, 0.9982857, -0.0025154, 0.0024831
6: 0.0035286, 0.0062696, 0.0035849, 0.0063276, -0.0022833, 0.0022539
7: -0.0102132, 0.0000154, -0.0100032, 0.0002318, -0.0085207, 0.0084111
8: -0.0092049, -0.0012439, -0.0093733, -0.0014074, -0.0065463, 0.0066317
9: -0.0039024, -0.0032156, -0.0038883, -0.0032011, -0.0005722, 0.0005648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016376, upper bound: 0.0016322
time: 1.93 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016376, upper bound: 0.0016338
time: 2.09 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0106362, -0.0037605, -0.0105587, -0.0035466, -0.0060345, 0.0052154
1: -0.0059374, -0.0039989, -0.0059156, -0.0039386, -0.0017014, 0.0014704
2: -0.0052477, 0.0090553, -0.0050864, 0.0095001, -0.0125530, 0.0108492
3: 0.0009329, 0.0028256, 0.0009542, 0.0028845, -0.0016612, 0.0014357
4: -0.0006755, 0.0100136, -0.0010079, 0.0098931, -0.0081080, 0.0093814
5: 0.9953185, 0.9982883, 0.9952263, 0.9982548, -0.0022526, 0.0026064
6: 0.0036343, 0.0063300, 0.0035505, 0.0062996, -0.0020447, 0.0023658
7: -0.0098188, 0.0002408, -0.0101317, 0.0001274, -0.0076305, 0.0088289
8: -0.0093803, -0.0015508, -0.0092920, -0.0013073, -0.0068715, 0.0059389
9: -0.0038759, -0.0032004, -0.0038969, -0.0032081, -0.0005124, 0.0005928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016577, upper bound: 0.0016502
time: 1.87 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016577, upper bound: 0.0016525
time: 2.10 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0105424, -0.0036823, -0.0105587, -0.0035466, -0.0058594, 0.0052472
1: -0.0059109, -0.0039768, -0.0059156, -0.0039386, -0.0016520, 0.0014794
2: -0.0050524, 0.0092179, -0.0050864, 0.0095001, -0.0121888, 0.0109152
3: 0.0009587, 0.0028471, 0.0009542, 0.0028845, -0.0016130, 0.0014444
4: -0.0007970, 0.0098677, -0.0010079, 0.0098931, -0.0081573, 0.0091091
5: 0.9952848, 0.9982478, 0.9952263, 0.9982548, -0.0022663, 0.0025308
6: 0.0036037, 0.0062932, 0.0035505, 0.0062996, -0.0020572, 0.0022972
7: -0.0099333, 0.0001035, -0.0101317, 0.0001274, -0.0076769, 0.0085727
8: -0.0092734, -0.0014618, -0.0092920, -0.0013073, -0.0066722, 0.0059750
9: -0.0038836, -0.0032097, -0.0038969, -0.0032081, -0.0005155, 0.0005756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016577, upper bound: 0.0016496
time: 2.03 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016577, upper bound: 0.0016524
time: 2.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.42 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0015974, upper bound: 0.0016027
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0015974, upper bound: 0.0016103
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016022, upper bound: 0.0016027
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016022, upper bound: 0.0016099
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016495, upper bound: 0.0016505
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016495, upper bound: 0.0016585
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016495, upper bound: 0.0016499
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016495, upper bound: 0.0016578
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016376, upper bound: 0.0016236
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016376, upper bound: 0.0016257
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016376, upper bound: 0.0016322
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016376, upper bound: 0.0016338
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016577, upper bound: 0.0016502
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016577, upper bound: 0.0016525
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016577, upper bound: 0.0016496
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 5, lower bound: -0.0016577, upper bound: 0.0016524

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0104926, -0.0037756, -0.0106003, -0.0038422, -0.0051412, 0.0052276
1: -0.0058969, -0.0040031, -0.0059273, -0.0040219, -0.0014495, 0.0014739
2: -0.0049489, 0.0090238, -0.0051730, 0.0088852, -0.0106947, 0.0108745
3: 0.0009724, 0.0028215, 0.0009427, 0.0028031, -0.0014153, 0.0014391
4: -0.0006520, 0.0097903, -0.0005484, 0.0099578, -0.0081269, 0.0079926
5: 0.9953251, 0.9982263, 0.9953539, 0.9982728, -0.0022579, 0.0022206
6: 0.0036402, 0.0062736, 0.0036664, 0.0063159, -0.0020495, 0.0020156
7: -0.0097967, 0.0000307, -0.0096992, 0.0001883, -0.0076483, 0.0075219
8: -0.0092167, -0.0015681, -0.0093394, -0.0016439, -0.0058543, 0.0059527
9: -0.0038745, -0.0032146, -0.0038679, -0.0032040, -0.0005136, 0.0005051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015789, upper bound: 0.0015757
time: 1.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015788, upper bound: 0.0015848
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0104926, -0.0037756, -0.0106328, -0.0037741, -0.0053031, 0.0053490
1: -0.0058969, -0.0040031, -0.0059365, -0.0040027, -0.0014952, 0.0015081
2: -0.0049489, 0.0090238, -0.0052406, 0.0090269, -0.0110316, 0.0111270
3: 0.0009724, 0.0028215, 0.0009338, 0.0028219, -0.0014599, 0.0014725
4: -0.0006520, 0.0097903, -0.0006543, 0.0100084, -0.0083156, 0.0082443
5: 0.9953251, 0.9982263, 0.9953244, 0.9982868, -0.0023103, 0.0022905
6: 0.0036402, 0.0062736, 0.0036397, 0.0063286, -0.0020971, 0.0020791
7: -0.0097967, 0.0000307, -0.0097989, 0.0002358, -0.0078259, 0.0077588
8: -0.0092167, -0.0015681, -0.0093764, -0.0015664, -0.0060387, 0.0060909
9: -0.0038745, -0.0032146, -0.0038746, -0.0032008, -0.0005255, 0.0005210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015789, upper bound: 0.0015823
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015788, upper bound: 0.0015920
time: 1.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106743, -0.0037516, -0.0105905, -0.0038862, -0.0053892, 0.0052926
1: -0.0059481, -0.0039964, -0.0059245, -0.0040343, -0.0015194, 0.0014922
2: -0.0053268, 0.0090737, -0.0051526, 0.0087938, -0.0112107, 0.0110096
3: 0.0009224, 0.0028281, 0.0009454, 0.0027910, -0.0014836, 0.0014569
4: -0.0006892, 0.0100728, -0.0004800, 0.0099426, -0.0082279, 0.0083782
5: 0.9953148, 0.9983047, 0.9953729, 0.9982685, -0.0022859, 0.0023277
6: 0.0036309, 0.0063449, 0.0036836, 0.0063120, -0.0020750, 0.0021129
7: -0.0098318, 0.0002965, -0.0096349, 0.0001740, -0.0077434, 0.0078848
8: -0.0094236, -0.0015408, -0.0093283, -0.0016940, -0.0061367, 0.0060267
9: -0.0038768, -0.0031967, -0.0038636, -0.0032049, -0.0005200, 0.0005294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015835, upper bound: 0.0015754
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015792, upper bound: 0.0015845
time: 2.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106743, -0.0037516, -0.0106230, -0.0038195, -0.0055427, 0.0054143
1: -0.0059481, -0.0039964, -0.0059337, -0.0040155, -0.0015627, 0.0015265
2: -0.0053268, 0.0090737, -0.0052202, 0.0089325, -0.0115300, 0.0112628
3: 0.0009224, 0.0028281, 0.0009365, 0.0028094, -0.0015258, 0.0014904
4: -0.0006892, 0.0100728, -0.0005837, 0.0099931, -0.0084171, 0.0086168
5: 0.9953148, 0.9983047, 0.9953440, 0.9982826, -0.0023385, 0.0023940
6: 0.0036309, 0.0063449, 0.0036575, 0.0063248, -0.0021227, 0.0021730
7: -0.0098318, 0.0002965, -0.0097325, 0.0002215, -0.0079214, 0.0081093
8: -0.0094236, -0.0015408, -0.0093653, -0.0016181, -0.0063115, 0.0061653
9: -0.0038768, -0.0031967, -0.0038701, -0.0032017, -0.0005319, 0.0005445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015835, upper bound: 0.0015820
time: 2.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015792, upper bound: 0.0015917
time: 2.22 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0106038, -0.0038284, -0.0105093, -0.0037568, -0.0053633, 0.0051745
1: -0.0059283, -0.0040180, -0.0059016, -0.0039978, -0.0015121, 0.0014589
2: -0.0051802, 0.0089141, -0.0049837, 0.0090629, -0.0111567, 0.0107640
3: 0.0009418, 0.0028069, 0.0009678, 0.0028266, -0.0014764, 0.0014244
4: -0.0005700, 0.0099632, -0.0006812, 0.0098164, -0.0080444, 0.0083379
5: 0.9953479, 0.9982744, 0.9953170, 0.9982336, -0.0022350, 0.0023165
6: 0.0036609, 0.0063173, 0.0036329, 0.0062802, -0.0020287, 0.0021027
7: -0.0097195, 0.0001934, -0.0098242, 0.0000552, -0.0075706, 0.0078468
8: -0.0093434, -0.0016281, -0.0092358, -0.0015467, -0.0061072, 0.0058922
9: -0.0038693, -0.0032036, -0.0038763, -0.0032129, -0.0005084, 0.0005269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016029, upper bound: 0.0016024
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016031, upper bound: 0.0016022
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0106038, -0.0038284, -0.0105424, -0.0036823, -0.0055052, 0.0052826
1: -0.0059283, -0.0040180, -0.0059109, -0.0039768, -0.0015521, 0.0014894
2: -0.0051802, 0.0089141, -0.0050524, 0.0092179, -0.0114520, 0.0109889
3: 0.0009418, 0.0028069, 0.0009587, 0.0028471, -0.0015155, 0.0014542
4: -0.0005700, 0.0099632, -0.0007970, 0.0098677, -0.0082124, 0.0085585
5: 0.9953479, 0.9982744, 0.9952848, 0.9982478, -0.0022817, 0.0023778
6: 0.0036609, 0.0063173, 0.0036037, 0.0062932, -0.0020711, 0.0021583
7: -0.0097195, 0.0001934, -0.0099333, 0.0001035, -0.0077288, 0.0080545
8: -0.0093434, -0.0016281, -0.0092734, -0.0014618, -0.0062689, 0.0060154
9: -0.0038693, -0.0032036, -0.0038836, -0.0032097, -0.0005190, 0.0005408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016029, upper bound: 0.0016105
time: 1.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016031, upper bound: 0.0016106
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0105093, -0.0037568, -0.0105093, -0.0037568, -0.0051985, 0.0051985
1: -0.0059016, -0.0039978, -0.0059016, -0.0039978, -0.0014657, 0.0014657
2: -0.0049837, 0.0090629, -0.0049837, 0.0090629, -0.0108140, 0.0108140
3: 0.0009678, 0.0028266, 0.0009678, 0.0028266, -0.0014311, 0.0014311
4: -0.0006812, 0.0098164, -0.0006812, 0.0098164, -0.0080817, 0.0080817
5: 0.9953170, 0.9982336, 0.9953170, 0.9982336, -0.0022453, 0.0022453
6: 0.0036329, 0.0062802, 0.0036329, 0.0062802, -0.0020381, 0.0020381
7: -0.0098242, 0.0000552, -0.0098242, 0.0000552, -0.0076058, 0.0076058
8: -0.0092358, -0.0015467, -0.0092358, -0.0015467, -0.0059196, 0.0059196
9: -0.0038763, -0.0032129, -0.0038763, -0.0032129, -0.0005107, 0.0005107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016017, upper bound: 0.0016058
time: 1.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016066, upper bound: 0.0016057
time: 1.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0105093, -0.0037568, -0.0105424, -0.0036823, -0.0053463, 0.0053201
1: -0.0059016, -0.0039978, -0.0059109, -0.0039768, -0.0015073, 0.0014999
2: -0.0049837, 0.0090629, -0.0050524, 0.0092179, -0.0111213, 0.0110668
3: 0.0009678, 0.0028266, 0.0009587, 0.0028471, -0.0014717, 0.0014645
4: -0.0006812, 0.0098164, -0.0007970, 0.0098677, -0.0082707, 0.0083114
5: 0.9953170, 0.9982336, 0.9952848, 0.9982478, -0.0022978, 0.0023092
6: 0.0036329, 0.0062802, 0.0036037, 0.0062932, -0.0020857, 0.0020960
7: -0.0098242, 0.0000552, -0.0099333, 0.0001035, -0.0077836, 0.0078219
8: -0.0092358, -0.0015467, -0.0092734, -0.0014618, -0.0060878, 0.0060580
9: -0.0038763, -0.0032129, -0.0038836, -0.0032097, -0.0005227, 0.0005252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016017, upper bound: 0.0016136
time: 1.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016066, upper bound: 0.0016138
time: 1.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0104474, -0.0036504, -0.0105752, -0.0038416, -0.0051889, 0.0057111
1: -0.0058842, -0.0039678, -0.0059202, -0.0040217, -0.0014629, 0.0016102
2: -0.0048550, 0.0092843, -0.0051207, 0.0088866, -0.0107939, 0.0118802
3: 0.0009848, 0.0028559, 0.0009496, 0.0028033, -0.0014284, 0.0015722
4: -0.0008466, 0.0097202, -0.0005494, 0.0099188, -0.0088785, 0.0080667
5: 0.9952711, 0.9982068, 0.9953536, 0.9982619, -0.0024667, 0.0022412
6: 0.0035912, 0.0062560, 0.0036661, 0.0063060, -0.0022390, 0.0020343
7: -0.0099799, -0.0000354, -0.0097002, 0.0001515, -0.0083557, 0.0075916
8: -0.0091653, -0.0014255, -0.0093108, -0.0016432, -0.0059086, 0.0065032
9: -0.0038868, -0.0032190, -0.0038680, -0.0032064, -0.0005611, 0.0005098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015879, upper bound: 0.0015760
time: 2.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015915, upper bound: 0.0015757
time: 1.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0104474, -0.0036504, -0.0106073, -0.0037736, -0.0051023, 0.0056083
1: -0.0058842, -0.0039678, -0.0059293, -0.0040026, -0.0014385, 0.0015812
2: -0.0048550, 0.0092843, -0.0051875, 0.0090281, -0.0106138, 0.0116664
3: 0.0009848, 0.0028559, 0.0009408, 0.0028220, -0.0014046, 0.0015439
4: -0.0008466, 0.0097202, -0.0006552, 0.0099687, -0.0087187, 0.0079321
5: 0.9952711, 0.9982068, 0.9953242, 0.9982758, -0.0024223, 0.0022038
6: 0.0035912, 0.0062560, 0.0036394, 0.0063186, -0.0021987, 0.0020004
7: -0.0099799, -0.0000354, -0.0097997, 0.0001985, -0.0082053, 0.0074650
8: -0.0091653, -0.0014255, -0.0093473, -0.0015657, -0.0058100, 0.0063862
9: -0.0038868, -0.0032190, -0.0038747, -0.0032033, -0.0005510, 0.0005013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015879, upper bound: 0.0015762
time: 2.10 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015915, upper bound: 0.0015758
time: 2.06 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0104822, -0.0034909, -0.0105802, -0.0038385, -0.0052204, 0.0058780
1: -0.0058940, -0.0039229, -0.0059216, -0.0040209, -0.0014718, 0.0016572
2: -0.0049272, 0.0096160, -0.0051312, 0.0088930, -0.0108595, 0.0122274
3: 0.0009753, 0.0028998, 0.0009483, 0.0028041, -0.0014371, 0.0016181
4: -0.0010946, 0.0097742, -0.0005542, 0.0099266, -0.0091380, 0.0081157
5: 0.9952021, 0.9982219, 0.9953522, 0.9982641, -0.0025388, 0.0022548
6: 0.0035286, 0.0062696, 0.0036649, 0.0063080, -0.0023045, 0.0020467
7: -0.0102132, 0.0000154, -0.0097047, 0.0001589, -0.0085999, 0.0076378
8: -0.0092049, -0.0012439, -0.0093165, -0.0016397, -0.0059445, 0.0066933
9: -0.0039024, -0.0032156, -0.0038683, -0.0032060, -0.0005775, 0.0005129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015878, upper bound: 0.0015850
time: 1.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015915, upper bound: 0.0015845
time: 1.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0104822, -0.0034909, -0.0106130, -0.0037705, -0.0051356, 0.0057880
1: -0.0058940, -0.0039229, -0.0059309, -0.0040017, -0.0014479, 0.0016319
2: -0.0049272, 0.0096160, -0.0051993, 0.0090344, -0.0106831, 0.0120403
3: 0.0009753, 0.0028998, 0.0009393, 0.0028229, -0.0014137, 0.0015933
4: -0.0010946, 0.0097742, -0.0006599, 0.0099775, -0.0089981, 0.0079839
5: 0.9952021, 0.9982219, 0.9953229, 0.9982783, -0.0024999, 0.0022182
6: 0.0035286, 0.0062696, 0.0036383, 0.0063208, -0.0022692, 0.0020134
7: -0.0102132, 0.0000154, -0.0098042, 0.0002068, -0.0084682, 0.0075137
8: -0.0092049, -0.0012439, -0.0093538, -0.0015623, -0.0058479, 0.0065909
9: -0.0039024, -0.0032156, -0.0038750, -0.0032027, -0.0005686, 0.0005045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015879, upper bound: 0.0015854
time: 2.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015915, upper bound: 0.0015849
time: 2.13 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0106362, -0.0037605, -0.0105093, -0.0037568, -0.0054845, 0.0053352
1: -0.0059374, -0.0039989, -0.0059016, -0.0039978, -0.0015463, 0.0015042
2: -0.0052477, 0.0090553, -0.0049837, 0.0090629, -0.0114089, 0.0110983
3: 0.0009329, 0.0028256, 0.0009678, 0.0028266, -0.0015098, 0.0014687
4: -0.0006755, 0.0100136, -0.0006812, 0.0098164, -0.0082942, 0.0085263
5: 0.9953185, 0.9982883, 0.9953170, 0.9982336, -0.0023044, 0.0023689
6: 0.0036343, 0.0063300, 0.0036329, 0.0062802, -0.0020917, 0.0021502
7: -0.0098188, 0.0002408, -0.0098242, 0.0000552, -0.0078057, 0.0080242
8: -0.0093803, -0.0015508, -0.0092358, -0.0015467, -0.0062453, 0.0060752
9: -0.0038759, -0.0032004, -0.0038763, -0.0032129, -0.0005241, 0.0005388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016399, upper bound: 0.0016201
time: 2.06 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016399, upper bound: 0.0016288
time: 1.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0106362, -0.0037605, -0.0105424, -0.0036823, -0.0053897, 0.0051991
1: -0.0059374, -0.0039989, -0.0059109, -0.0039768, -0.0015196, 0.0014658
2: -0.0052477, 0.0090553, -0.0050524, 0.0092179, -0.0112117, 0.0108151
3: 0.0009329, 0.0028256, 0.0009587, 0.0028471, -0.0014837, 0.0014312
4: -0.0006755, 0.0100136, -0.0007970, 0.0098677, -0.0080825, 0.0083789
5: 0.9953185, 0.9982883, 0.9952848, 0.9982478, -0.0022456, 0.0023279
6: 0.0036343, 0.0063300, 0.0036037, 0.0062932, -0.0020383, 0.0021130
7: -0.0098188, 0.0002408, -0.0099333, 0.0001035, -0.0076066, 0.0078855
8: -0.0093803, -0.0015508, -0.0092734, -0.0014618, -0.0061373, 0.0059202
9: -0.0038759, -0.0032004, -0.0038836, -0.0032097, -0.0005108, 0.0005295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016399, upper bound: 0.0016221
time: 2.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016399, upper bound: 0.0016308
time: 2.28 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0105424, -0.0036823, -0.0105093, -0.0037568, -0.0053201, 0.0053463
1: -0.0059109, -0.0039768, -0.0059016, -0.0039978, -0.0014999, 0.0015073
2: -0.0050524, 0.0092179, -0.0049837, 0.0090629, -0.0110668, 0.0111213
3: 0.0009587, 0.0028471, 0.0009678, 0.0028266, -0.0014645, 0.0014717
4: -0.0007970, 0.0098677, -0.0006812, 0.0098164, -0.0083114, 0.0082707
5: 0.9952848, 0.9982478, 0.9953170, 0.9982336, -0.0023092, 0.0022978
6: 0.0036037, 0.0062932, 0.0036329, 0.0062802, -0.0020960, 0.0020857
7: -0.0099333, 0.0001035, -0.0098242, 0.0000552, -0.0078220, 0.0077836
8: -0.0092734, -0.0014618, -0.0092358, -0.0015467, -0.0060580, 0.0060878
9: -0.0038836, -0.0032097, -0.0038763, -0.0032129, -0.0005252, 0.0005227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016378, upper bound: 0.0016193
time: 2.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016378, upper bound: 0.0016290
time: 1.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0105424, -0.0036823, -0.0105424, -0.0036823, -0.0052306, 0.0052306
1: -0.0059109, -0.0039768, -0.0059109, -0.0039768, -0.0014747, 0.0014747
2: -0.0050524, 0.0092179, -0.0050524, 0.0092179, -0.0108808, 0.0108808
3: 0.0009587, 0.0028471, 0.0009587, 0.0028471, -0.0014399, 0.0014399
4: -0.0007970, 0.0098677, -0.0007970, 0.0098677, -0.0081316, 0.0081316
5: 0.9952848, 0.9982478, 0.9952848, 0.9982478, -0.0022592, 0.0022592
6: 0.0036037, 0.0062932, 0.0036037, 0.0062932, -0.0020507, 0.0020507
7: -0.0099333, 0.0001035, -0.0099333, 0.0001035, -0.0076528, 0.0076528
8: -0.0092734, -0.0014618, -0.0092734, -0.0014618, -0.0059562, 0.0059562
9: -0.0038836, -0.0032097, -0.0038836, -0.0032097, -0.0005139, 0.0005139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016378, upper bound: 0.0016224
time: 2.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016378, upper bound: 0.0016307
time: 2.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.52 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015789, upper bound: 0.0015757
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015788, upper bound: 0.0015848
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015789, upper bound: 0.0015823
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015788, upper bound: 0.0015920
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015835, upper bound: 0.0015754
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015792, upper bound: 0.0015845
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015835, upper bound: 0.0015820
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015792, upper bound: 0.0015917
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016029, upper bound: 0.0016024
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016031, upper bound: 0.0016022
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016029, upper bound: 0.0016105
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016031, upper bound: 0.0016106
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016017, upper bound: 0.0016058
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016066, upper bound: 0.0016057
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016017, upper bound: 0.0016136
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016066, upper bound: 0.0016138
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015879, upper bound: 0.0015760
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015915, upper bound: 0.0015757
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015879, upper bound: 0.0015762
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015915, upper bound: 0.0015758
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015878, upper bound: 0.0015850
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015915, upper bound: 0.0015845
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015879, upper bound: 0.0015854
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0015915, upper bound: 0.0015849
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016399, upper bound: 0.0016201
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016399, upper bound: 0.0016288
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016399, upper bound: 0.0016221
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016399, upper bound: 0.0016308
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016378, upper bound: 0.0016193
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016378, upper bound: 0.0016290
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016378, upper bound: 0.0016224
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 5, lower bound: -0.0016378, upper bound: 0.0016307

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0103903, -0.0038250, -0.0105717, -0.0038555, -0.0050365, 0.0051540
1: -0.0058681, -0.0040171, -0.0059192, -0.0040257, -0.0014200, 0.0014531
2: -0.0047362, 0.0089211, -0.0051135, 0.0088577, -0.0104770, 0.0107213
3: 0.0010005, 0.0028079, 0.0009506, 0.0027995, -0.0013865, 0.0014188
4: -0.0005752, 0.0096314, -0.0005278, 0.0099134, -0.0080125, 0.0078299
5: 0.9953465, 0.9981821, 0.9953596, 0.9982605, -0.0022261, 0.0021754
6: 0.0036596, 0.0062336, 0.0036716, 0.0063047, -0.0020206, 0.0019746
7: -0.0097244, -0.0001189, -0.0096799, 0.0001465, -0.0075406, 0.0073688
8: -0.0091003, -0.0016243, -0.0093069, -0.0016590, -0.0057351, 0.0058689
9: -0.0038696, -0.0032246, -0.0038666, -0.0032068, -0.0005063, 0.0004948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015421, upper bound: 0.0015412
time: 1.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015475, upper bound: 0.0015418
time: 1.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0104236, -0.0036743, -0.0105768, -0.0038524, -0.0050695, 0.0053412
1: -0.0058775, -0.0039746, -0.0059206, -0.0040248, -0.0014293, 0.0015059
2: -0.0048054, 0.0092346, -0.0051240, 0.0088641, -0.0105456, 0.0111107
3: 0.0009914, 0.0028493, 0.0009492, 0.0028003, -0.0013955, 0.0014703
4: -0.0008095, 0.0096831, -0.0005326, 0.0099212, -0.0083035, 0.0078811
5: 0.9952813, 0.9981965, 0.9953583, 0.9982627, -0.0023070, 0.0021896
6: 0.0036005, 0.0062466, 0.0036703, 0.0063067, -0.0020940, 0.0019875
7: -0.0099449, -0.0000702, -0.0096844, 0.0001538, -0.0078145, 0.0074170
8: -0.0091382, -0.0014527, -0.0093126, -0.0016555, -0.0057727, 0.0060820
9: -0.0038844, -0.0032213, -0.0038669, -0.0032063, -0.0005247, 0.0004980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015420, upper bound: 0.0015528
time: 2.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015475, upper bound: 0.0015528
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0103903, -0.0038250, -0.0106039, -0.0037872, -0.0051991, 0.0052733
1: -0.0058681, -0.0040171, -0.0059283, -0.0040064, -0.0014658, 0.0014868
2: -0.0047362, 0.0089211, -0.0051804, 0.0089997, -0.0108151, 0.0109696
3: 0.0010005, 0.0028079, 0.0009417, 0.0028183, -0.0014312, 0.0014517
4: -0.0005752, 0.0096314, -0.0006340, 0.0099634, -0.0081980, 0.0080826
5: 0.9953465, 0.9981821, 0.9953301, 0.9982743, -0.0022777, 0.0022456
6: 0.0036596, 0.0062336, 0.0036448, 0.0063173, -0.0020674, 0.0020383
7: -0.0097244, -0.0001189, -0.0097798, 0.0001935, -0.0077153, 0.0076066
8: -0.0091003, -0.0016243, -0.0093435, -0.0015813, -0.0059202, 0.0060048
9: -0.0038696, -0.0032246, -0.0038733, -0.0032036, -0.0005181, 0.0005108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015420, upper bound: 0.0015486
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015474, upper bound: 0.0015486
time: 2.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0104236, -0.0036743, -0.0106096, -0.0037842, -0.0052318, 0.0054642
1: -0.0058775, -0.0039746, -0.0059299, -0.0040056, -0.0014750, 0.0015406
2: -0.0048054, 0.0092346, -0.0051922, 0.0090060, -0.0108831, 0.0113667
3: 0.0009914, 0.0028493, 0.0009402, 0.0028191, -0.0014402, 0.0015042
4: -0.0008095, 0.0096831, -0.0006387, 0.0099722, -0.0084948, 0.0081334
5: 0.9952813, 0.9981965, 0.9953288, 0.9982769, -0.0023601, 0.0022597
6: 0.0036005, 0.0062466, 0.0036436, 0.0063195, -0.0021423, 0.0020511
7: -0.0099449, -0.0000702, -0.0097842, 0.0002018, -0.0079945, 0.0076544
8: -0.0091382, -0.0014527, -0.0093499, -0.0015778, -0.0059574, 0.0062222
9: -0.0038844, -0.0032213, -0.0038736, -0.0032031, -0.0005368, 0.0005140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015423, upper bound: 0.0015595
time: 2.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015474, upper bound: 0.0015598
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0105728, -0.0038016, -0.0105619, -0.0038994, -0.0052875, 0.0052172
1: -0.0059195, -0.0040105, -0.0059164, -0.0040381, -0.0014907, 0.0014709
2: -0.0051158, 0.0089698, -0.0050930, 0.0087662, -0.0109990, 0.0108529
3: 0.0009503, 0.0028143, 0.0009533, 0.0027874, -0.0014555, 0.0014362
4: -0.0006116, 0.0099151, -0.0004594, 0.0098981, -0.0081108, 0.0082199
5: 0.9953364, 0.9982610, 0.9953786, 0.9982562, -0.0022534, 0.0022837
6: 0.0036504, 0.0063051, 0.0036888, 0.0063008, -0.0020454, 0.0020730
7: -0.0097587, 0.0001481, -0.0096155, 0.0001320, -0.0076332, 0.0077359
8: -0.0093081, -0.0015976, -0.0092956, -0.0017091, -0.0060209, 0.0059409
9: -0.0038719, -0.0032067, -0.0038623, -0.0032078, -0.0005126, 0.0005194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015493, upper bound: 0.0015415
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015513, upper bound: 0.0015418
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0106053, -0.0036615, -0.0105669, -0.0038964, -0.0053159, 0.0053888
1: -0.0059287, -0.0039710, -0.0059179, -0.0040372, -0.0014988, 0.0015193
2: -0.0051832, 0.0092611, -0.0051035, 0.0087726, -0.0110582, 0.0112098
3: 0.0009414, 0.0028529, 0.0009519, 0.0027882, -0.0014634, 0.0014834
4: -0.0008293, 0.0099655, -0.0004643, 0.0099059, -0.0083775, 0.0082642
5: 0.9952759, 0.9982750, 0.9953772, 0.9982584, -0.0023275, 0.0022960
6: 0.0035955, 0.0063178, 0.0036876, 0.0063028, -0.0021127, 0.0020841
7: -0.0099636, 0.0001955, -0.0096201, 0.0001394, -0.0078842, 0.0077775
8: -0.0093450, -0.0014382, -0.0093014, -0.0017056, -0.0060533, 0.0061363
9: -0.0038857, -0.0032035, -0.0038626, -0.0032073, -0.0005294, 0.0005222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015494, upper bound: 0.0015522
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015514, upper bound: 0.0015528
time: 1.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0105728, -0.0038016, -0.0105941, -0.0038326, -0.0054414, 0.0053369
1: -0.0059195, -0.0040105, -0.0059255, -0.0040192, -0.0015341, 0.0015047
2: -0.0051158, 0.0089698, -0.0051600, 0.0089052, -0.0113193, 0.0111019
3: 0.0009503, 0.0028143, 0.0009445, 0.0028058, -0.0014979, 0.0014692
4: -0.0006116, 0.0099151, -0.0005634, 0.0099481, -0.0082968, 0.0084593
5: 0.9953364, 0.9982610, 0.9953497, 0.9982701, -0.0023051, 0.0023503
6: 0.0036504, 0.0063051, 0.0036626, 0.0063134, -0.0020923, 0.0021333
7: -0.0097587, 0.0001481, -0.0097133, 0.0001791, -0.0078083, 0.0079612
8: -0.0093081, -0.0015976, -0.0093323, -0.0016330, -0.0061962, 0.0060772
9: -0.0038719, -0.0032067, -0.0038689, -0.0032046, -0.0005243, 0.0005346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015494, upper bound: 0.0015483
time: 2.17 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015514, upper bound: 0.0015486
time: 1.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0106053, -0.0036615, -0.0105998, -0.0038296, -0.0054697, 0.0055122
1: -0.0059287, -0.0039710, -0.0059271, -0.0040184, -0.0015421, 0.0015541
2: -0.0051832, 0.0092611, -0.0051719, 0.0089115, -0.0113780, 0.0114665
3: 0.0009414, 0.0028529, 0.0009429, 0.0028066, -0.0015057, 0.0015174
4: -0.0008293, 0.0099655, -0.0005681, 0.0099570, -0.0085693, 0.0085032
5: 0.9952759, 0.9982750, 0.9953484, 0.9982726, -0.0023808, 0.0023624
6: 0.0035955, 0.0063178, 0.0036614, 0.0063157, -0.0021611, 0.0021444
7: -0.0099636, 0.0001955, -0.0097177, 0.0001875, -0.0080647, 0.0080025
8: -0.0093450, -0.0014382, -0.0093388, -0.0016295, -0.0062283, 0.0062768
9: -0.0038857, -0.0032035, -0.0038691, -0.0032040, -0.0005415, 0.0005373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015496, upper bound: 0.0015593
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015513, upper bound: 0.0015595
time: 2.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0105784, -0.0039262, -0.0105059, -0.0037715, -0.0053202, 0.0050639
1: -0.0059211, -0.0040456, -0.0059007, -0.0040020, -0.0015000, 0.0014277
2: -0.0051273, 0.0087105, -0.0049766, 0.0090324, -0.0110671, 0.0105340
3: 0.0009488, 0.0027800, 0.0009687, 0.0028226, -0.0014646, 0.0013940
4: -0.0004178, 0.0099237, -0.0006584, 0.0098111, -0.0078724, 0.0082709
5: 0.9953901, 0.9982634, 0.9953234, 0.9982321, -0.0021872, 0.0022979
6: 0.0036993, 0.0063073, 0.0036386, 0.0062789, -0.0019853, 0.0020858
7: -0.0095763, 0.0001562, -0.0098028, 0.0000502, -0.0074088, 0.0077838
8: -0.0093144, -0.0017396, -0.0092319, -0.0015634, -0.0060581, 0.0057663
9: -0.0038597, -0.0032061, -0.0038749, -0.0032133, -0.0004975, 0.0005227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015741, upper bound: 0.0015835
time: 1.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015851, upper bound: 0.0015835
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0107380, -0.0039141, -0.0104964, -0.0038130, -0.0055327, 0.0051123
1: -0.0059661, -0.0040422, -0.0058980, -0.0040137, -0.0015599, 0.0014414
2: -0.0054593, 0.0087356, -0.0049568, 0.0089460, -0.0115092, 0.0106347
3: 0.0009048, 0.0027833, 0.0009713, 0.0028112, -0.0015231, 0.0014073
4: -0.0004366, 0.0101718, -0.0005938, 0.0097963, -0.0079477, 0.0086013
5: 0.9953850, 0.9983323, 0.9953413, 0.9982280, -0.0022081, 0.0023897
6: 0.0036946, 0.0063699, 0.0036549, 0.0062751, -0.0020043, 0.0021691
7: -0.0095940, 0.0003897, -0.0097420, 0.0000362, -0.0074797, 0.0080947
8: -0.0094962, -0.0017258, -0.0092211, -0.0016107, -0.0063002, 0.0058215
9: -0.0038608, -0.0031905, -0.0038708, -0.0032142, -0.0005022, 0.0005435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015761, upper bound: 0.0015832
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015849, upper bound: 0.0015838
time: 1.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0105784, -0.0039262, -0.0105390, -0.0036968, -0.0054631, 0.0051721
1: -0.0059211, -0.0040456, -0.0059100, -0.0039809, -0.0015403, 0.0014582
2: -0.0051273, 0.0087105, -0.0050454, 0.0091877, -0.0113644, 0.0107591
3: 0.0009488, 0.0027800, 0.0009596, 0.0028431, -0.0015039, 0.0014238
4: -0.0004178, 0.0099237, -0.0007745, 0.0098625, -0.0080407, 0.0084930
5: 0.9953901, 0.9982634, 0.9952911, 0.9982463, -0.0022339, 0.0023596
6: 0.0036993, 0.0063073, 0.0036094, 0.0062919, -0.0020277, 0.0021418
7: -0.0095763, 0.0001562, -0.0099120, 0.0000986, -0.0075672, 0.0079929
8: -0.0093144, -0.0017396, -0.0092696, -0.0014783, -0.0062209, 0.0058895
9: -0.0038597, -0.0032061, -0.0038822, -0.0032100, -0.0005081, 0.0005367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015741, upper bound: 0.0015917
time: 1.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015851, upper bound: 0.0015918
time: 1.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0107380, -0.0039141, -0.0105297, -0.0037387, -0.0056703, 0.0052210
1: -0.0059661, -0.0040422, -0.0059074, -0.0039927, -0.0015987, 0.0014720
2: -0.0054593, 0.0087356, -0.0050260, 0.0091005, -0.0117953, 0.0108607
3: 0.0009048, 0.0027833, 0.0009622, 0.0028316, -0.0015609, 0.0014372
4: -0.0004366, 0.0101718, -0.0007093, 0.0098480, -0.0081166, 0.0088151
5: 0.9953850, 0.9983323, 0.9953091, 0.9982423, -0.0022550, 0.0024491
6: 0.0036946, 0.0063699, 0.0036258, 0.0062882, -0.0020469, 0.0022230
7: -0.0095940, 0.0003897, -0.0098507, 0.0000849, -0.0076387, 0.0082960
8: -0.0094962, -0.0017258, -0.0092589, -0.0015261, -0.0064568, 0.0059452
9: -0.0038608, -0.0031905, -0.0038781, -0.0032109, -0.0005129, 0.0005571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015758, upper bound: 0.0015915
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015849, upper bound: 0.0015915
time: 1.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0104844, -0.0038612, -0.0105059, -0.0037715, -0.0051557, 0.0050751
1: -0.0058946, -0.0040273, -0.0059007, -0.0040020, -0.0014536, 0.0014309
2: -0.0049319, 0.0088457, -0.0049766, 0.0090324, -0.0107250, 0.0105572
3: 0.0009746, 0.0027979, 0.0009687, 0.0028226, -0.0014193, 0.0013971
4: -0.0005188, 0.0097777, -0.0006584, 0.0098111, -0.0078898, 0.0080152
5: 0.9953622, 0.9982228, 0.9953234, 0.9982321, -0.0021920, 0.0022269
6: 0.0036738, 0.0062705, 0.0036386, 0.0062789, -0.0019897, 0.0020213
7: -0.0096714, 0.0000187, -0.0098028, 0.0000502, -0.0074252, 0.0075432
8: -0.0092075, -0.0016656, -0.0092319, -0.0015634, -0.0058709, 0.0057790
9: -0.0038660, -0.0032154, -0.0038749, -0.0032133, -0.0004986, 0.0005065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015723, upper bound: 0.0015861
time: 1.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015822, upper bound: 0.0015863
time: 2.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0106661, -0.0038276, -0.0104964, -0.0038130, -0.0053931, 0.0051442
1: -0.0059458, -0.0040178, -0.0058980, -0.0040137, -0.0015205, 0.0014503
2: -0.0053098, 0.0089156, -0.0049568, 0.0089460, -0.0112188, 0.0107009
3: 0.0009246, 0.0028071, 0.0009713, 0.0028112, -0.0014846, 0.0014161
4: -0.0005711, 0.0100601, -0.0005938, 0.0097963, -0.0079972, 0.0083843
5: 0.9953476, 0.9983013, 0.9953413, 0.9982280, -0.0022219, 0.0023294
6: 0.0036606, 0.0063417, 0.0036549, 0.0062751, -0.0020168, 0.0021144
7: -0.0097206, 0.0002845, -0.0097420, 0.0000362, -0.0075262, 0.0078905
8: -0.0094143, -0.0016273, -0.0092211, -0.0016107, -0.0061412, 0.0058577
9: -0.0038693, -0.0031975, -0.0038708, -0.0032142, -0.0005054, 0.0005298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015776, upper bound: 0.0015860
time: 1.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015868, upper bound: 0.0015863
time: 1.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0104844, -0.0038612, -0.0105390, -0.0036968, -0.0053046, 0.0051967
1: -0.0058946, -0.0040273, -0.0059100, -0.0039809, -0.0014956, 0.0014651
2: -0.0049319, 0.0088457, -0.0050454, 0.0091877, -0.0110346, 0.0108102
3: 0.0009746, 0.0027979, 0.0009596, 0.0028431, -0.0014603, 0.0014306
4: -0.0005188, 0.0097777, -0.0007745, 0.0098625, -0.0080789, 0.0082466
5: 0.9953622, 0.9982228, 0.9952911, 0.9982463, -0.0022445, 0.0022911
6: 0.0036738, 0.0062705, 0.0036094, 0.0062919, -0.0020374, 0.0020797
7: -0.0096714, 0.0000187, -0.0099120, 0.0000986, -0.0076031, 0.0077609
8: -0.0092075, -0.0016656, -0.0092696, -0.0014783, -0.0060403, 0.0059175
9: -0.0038660, -0.0032154, -0.0038822, -0.0032100, -0.0005105, 0.0005211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015723, upper bound: 0.0015937
time: 2.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015822, upper bound: 0.0015934
time: 2.13 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0106661, -0.0038276, -0.0105297, -0.0037387, -0.0055382, 0.0052662
1: -0.0059458, -0.0040178, -0.0059074, -0.0039927, -0.0015614, 0.0014847
2: -0.0053098, 0.0089156, -0.0050260, 0.0091005, -0.0115205, 0.0109547
3: 0.0009246, 0.0028071, 0.0009622, 0.0028316, -0.0015246, 0.0014497
4: -0.0005711, 0.0100601, -0.0007093, 0.0098480, -0.0081869, 0.0086097
5: 0.9953476, 0.9983013, 0.9953091, 0.9982423, -0.0022746, 0.0023920
6: 0.0036606, 0.0063417, 0.0036258, 0.0062882, -0.0020646, 0.0021712
7: -0.0097206, 0.0002845, -0.0098507, 0.0000849, -0.0077047, 0.0081027
8: -0.0094143, -0.0016273, -0.0092589, -0.0015261, -0.0063063, 0.0059966
9: -0.0038693, -0.0031975, -0.0038781, -0.0032109, -0.0005174, 0.0005441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015776, upper bound: 0.0015937
time: 2.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015868, upper bound: 0.0015939
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0104228, -0.0037500, -0.0105717, -0.0038555, -0.0051483, 0.0055966
1: -0.0058772, -0.0039959, -0.0059192, -0.0040257, -0.0014515, 0.0015779
2: -0.0048036, 0.0090772, -0.0051135, 0.0088577, -0.0107095, 0.0116420
3: 0.0009916, 0.0028285, 0.0009506, 0.0027995, -0.0014172, 0.0015406
4: -0.0006918, 0.0096818, -0.0005278, 0.0099134, -0.0087005, 0.0080036
5: 0.9953141, 0.9981961, 0.9953596, 0.9982605, -0.0024173, 0.0022236
6: 0.0036302, 0.0062463, 0.0036716, 0.0063047, -0.0021941, 0.0020184
7: -0.0098342, -0.0000715, -0.0096799, 0.0001465, -0.0081881, 0.0075323
8: -0.0091372, -0.0015389, -0.0093069, -0.0016590, -0.0058624, 0.0063728
9: -0.0038770, -0.0032214, -0.0038666, -0.0032068, -0.0005498, 0.0005058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015497, upper bound: 0.0015412
time: 1.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015566, upper bound: 0.0015415
time: 1.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0106049, -0.0037304, -0.0105619, -0.0038994, -0.0053725, 0.0056336
1: -0.0059286, -0.0039904, -0.0059164, -0.0040381, -0.0015147, 0.0015883
2: -0.0051824, 0.0091179, -0.0050930, 0.0087662, -0.0111759, 0.0117191
3: 0.0009415, 0.0028339, 0.0009533, 0.0027874, -0.0014790, 0.0015508
4: -0.0007222, 0.0099649, -0.0004594, 0.0098981, -0.0087581, 0.0083522
5: 0.9953057, 0.9982747, 0.9953786, 0.9982562, -0.0024333, 0.0023205
6: 0.0036225, 0.0063177, 0.0036888, 0.0063008, -0.0022087, 0.0021063
7: -0.0098628, 0.0001949, -0.0096155, 0.0001320, -0.0082423, 0.0078603
8: -0.0093446, -0.0015166, -0.0092956, -0.0017091, -0.0061177, 0.0064150
9: -0.0038789, -0.0032035, -0.0038623, -0.0032078, -0.0005535, 0.0005278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015564, upper bound: 0.0015418
time: 2.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015597, upper bound: 0.0015415
time: 1.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0104228, -0.0037500, -0.0106039, -0.0037872, -0.0050612, 0.0054892
1: -0.0058772, -0.0039959, -0.0059283, -0.0040064, -0.0014269, 0.0015476
2: -0.0048036, 0.0090772, -0.0051804, 0.0089997, -0.0105284, 0.0114187
3: 0.0009916, 0.0028285, 0.0009417, 0.0028183, -0.0013933, 0.0015111
4: -0.0006918, 0.0096818, -0.0006340, 0.0099634, -0.0085336, 0.0078683
5: 0.9953141, 0.9981961, 0.9953301, 0.9982743, -0.0023709, 0.0021860
6: 0.0036302, 0.0062463, 0.0036448, 0.0063173, -0.0021521, 0.0019843
7: -0.0098342, -0.0000715, -0.0097798, 0.0001935, -0.0080311, 0.0074049
8: -0.0091372, -0.0015389, -0.0093435, -0.0015813, -0.0057633, 0.0062506
9: -0.0038770, -0.0032214, -0.0038733, -0.0032036, -0.0005393, 0.0004972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015495, upper bound: 0.0015414
time: 1.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015566, upper bound: 0.0015417
time: 2.08 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0106049, -0.0037304, -0.0105941, -0.0038326, -0.0053116, 0.0055334
1: -0.0059286, -0.0039904, -0.0059255, -0.0040192, -0.0014975, 0.0015601
2: -0.0051824, 0.0091179, -0.0051600, 0.0089052, -0.0110492, 0.0115106
3: 0.0009415, 0.0028339, 0.0009445, 0.0028058, -0.0014622, 0.0015232
4: -0.0007222, 0.0099649, -0.0005634, 0.0099481, -0.0086023, 0.0082575
5: 0.9953057, 0.9982747, 0.9953497, 0.9982701, -0.0023900, 0.0022942
6: 0.0036225, 0.0063177, 0.0036626, 0.0063134, -0.0021694, 0.0020824
7: -0.0098628, 0.0001949, -0.0097133, 0.0001791, -0.0080957, 0.0077712
8: -0.0093446, -0.0015166, -0.0093323, -0.0016330, -0.0060483, 0.0063009
9: -0.0038789, -0.0032035, -0.0038689, -0.0032046, -0.0005436, 0.0005218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015565, upper bound: 0.0015420
time: 1.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015598, upper bound: 0.0015417
time: 2.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0104573, -0.0035864, -0.0105768, -0.0038524, -0.0051796, 0.0057654
1: -0.0058870, -0.0039498, -0.0059206, -0.0040248, -0.0014603, 0.0016255
2: -0.0048755, 0.0094175, -0.0051240, 0.0088641, -0.0107746, 0.0119931
3: 0.0009821, 0.0028735, 0.0009492, 0.0028003, -0.0014258, 0.0015871
4: -0.0009462, 0.0097355, -0.0005326, 0.0099212, -0.0089629, 0.0080523
5: 0.9952434, 0.9982110, 0.9953583, 0.9982627, -0.0024902, 0.0022372
6: 0.0035661, 0.0062598, 0.0036703, 0.0063067, -0.0022603, 0.0020307
7: -0.0100736, -0.0000210, -0.0096844, 0.0001538, -0.0084351, 0.0075781
8: -0.0091766, -0.0013526, -0.0093126, -0.0016555, -0.0058980, 0.0065651
9: -0.0038930, -0.0032180, -0.0038669, -0.0032063, -0.0005664, 0.0005089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015494, upper bound: 0.0015528
time: 2.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015566, upper bound: 0.0015523
time: 1.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0106348, -0.0035810, -0.0105669, -0.0038964, -0.0054042, 0.0058015
1: -0.0059370, -0.0039483, -0.0059179, -0.0040372, -0.0015236, 0.0016357
2: -0.0052447, 0.0094287, -0.0051035, 0.0087726, -0.0112418, 0.0120683
3: 0.0009332, 0.0028750, 0.0009519, 0.0027882, -0.0014877, 0.0015971
4: -0.0009546, 0.0100114, -0.0004643, 0.0099059, -0.0090191, 0.0084014
5: 0.9952410, 0.9982877, 0.9953772, 0.9982584, -0.0025058, 0.0023342
6: 0.0035639, 0.0063294, 0.0036876, 0.0063028, -0.0022745, 0.0021187
7: -0.0100815, 0.0002387, -0.0096201, 0.0001394, -0.0084880, 0.0079067
8: -0.0093787, -0.0013464, -0.0093014, -0.0017056, -0.0061538, 0.0066062
9: -0.0038936, -0.0032006, -0.0038626, -0.0032073, -0.0005700, 0.0005309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015563, upper bound: 0.0015528
time: 2.01 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015597, upper bound: 0.0015526
time: 1.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0104573, -0.0035864, -0.0106096, -0.0037842, -0.0050943, 0.0056783
1: -0.0058870, -0.0039498, -0.0059299, -0.0040056, -0.0014363, 0.0016009
2: -0.0048755, 0.0094175, -0.0051922, 0.0090060, -0.0105972, 0.0118121
3: 0.0009821, 0.0028735, 0.0009402, 0.0028191, -0.0014024, 0.0015631
4: -0.0009462, 0.0097355, -0.0006387, 0.0099722, -0.0088276, 0.0079197
5: 0.9952434, 0.9982110, 0.9953288, 0.9982769, -0.0024526, 0.0022003
6: 0.0035661, 0.0062598, 0.0036436, 0.0063195, -0.0022262, 0.0019972
7: -0.0100736, -0.0000210, -0.0097842, 0.0002018, -0.0083078, 0.0074533
8: -0.0091766, -0.0013526, -0.0093499, -0.0015778, -0.0058009, 0.0064660
9: -0.0038930, -0.0032180, -0.0038736, -0.0032031, -0.0005579, 0.0005005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015495, upper bound: 0.0015530
time: 1.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015567, upper bound: 0.0015527
time: 2.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0106348, -0.0035810, -0.0105998, -0.0038296, -0.0053443, 0.0057160
1: -0.0059370, -0.0039483, -0.0059271, -0.0040184, -0.0015068, 0.0016115
2: -0.0052447, 0.0094287, -0.0051719, 0.0089115, -0.0111173, 0.0118904
3: 0.0009332, 0.0028750, 0.0009429, 0.0028066, -0.0014712, 0.0015735
4: -0.0009546, 0.0100114, -0.0005681, 0.0099570, -0.0088861, 0.0083083
5: 0.9952410, 0.9982877, 0.9953484, 0.9982726, -0.0024688, 0.0023083
6: 0.0035639, 0.0063294, 0.0036614, 0.0063157, -0.0022409, 0.0020952
7: -0.0100815, 0.0002387, -0.0097177, 0.0001875, -0.0083628, 0.0078191
8: -0.0093787, -0.0013464, -0.0093388, -0.0016295, -0.0060856, 0.0065088
9: -0.0038936, -0.0032006, -0.0038691, -0.0032040, -0.0005615, 0.0005250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015565, upper bound: 0.0015528
time: 1.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015598, upper bound: 0.0015525
time: 2.26 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0105325, -0.0038088, -0.0104812, -0.0037705, -0.0053746, 0.0052632
1: -0.0059082, -0.0040125, -0.0058937, -0.0040017, -0.0015153, 0.0014839
2: -0.0050320, 0.0089547, -0.0049253, 0.0090345, -0.0111803, 0.0109486
3: 0.0009614, 0.0028123, 0.0009755, 0.0028229, -0.0014795, 0.0014489
4: -0.0006003, 0.0098525, -0.0006600, 0.0097727, -0.0081823, 0.0083555
5: 0.9953395, 0.9982436, 0.9953228, 0.9982213, -0.0022733, 0.0023214
6: 0.0036533, 0.0062893, 0.0036382, 0.0062692, -0.0020635, 0.0021071
7: -0.0097481, 0.0000891, -0.0098042, 0.0000141, -0.0077004, 0.0078634
8: -0.0092622, -0.0016059, -0.0092038, -0.0015622, -0.0061201, 0.0059933
9: -0.0038712, -0.0032106, -0.0038750, -0.0032157, -0.0005171, 0.0005280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015690
time: 1.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015742
time: 1.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0105700, -0.0036568, -0.0104850, -0.0037673, -0.0054097, 0.0054339
1: -0.0059187, -0.0039696, -0.0058948, -0.0040008, -0.0015252, 0.0015320
2: -0.0051098, 0.0092710, -0.0049330, 0.0090412, -0.0112533, 0.0113037
3: 0.0009511, 0.0028542, 0.0009745, 0.0028237, -0.0014892, 0.0014959
4: -0.0008367, 0.0099106, -0.0006649, 0.0097785, -0.0084476, 0.0084100
5: 0.9952738, 0.9982597, 0.9953215, 0.9982231, -0.0023470, 0.0023365
6: 0.0035937, 0.0063040, 0.0036370, 0.0062707, -0.0021304, 0.0021209
7: -0.0099706, 0.0001439, -0.0098089, 0.0000195, -0.0079502, 0.0079147
8: -0.0093048, -0.0014328, -0.0092081, -0.0015586, -0.0061601, 0.0061876
9: -0.0038861, -0.0032070, -0.0038753, -0.0032153, -0.0005338, 0.0005315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015917, upper bound: 0.0015791
time: 1.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015920, upper bound: 0.0015831
time: 2.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0105325, -0.0038088, -0.0105141, -0.0036957, -0.0052883, 0.0051254
1: -0.0059082, -0.0040125, -0.0059030, -0.0039806, -0.0014910, 0.0014450
2: -0.0050320, 0.0089547, -0.0049936, 0.0091901, -0.0110007, 0.0106619
3: 0.0009614, 0.0028123, 0.0009665, 0.0028435, -0.0014558, 0.0014109
4: -0.0006003, 0.0098525, -0.0007763, 0.0098238, -0.0079680, 0.0082212
5: 0.9953395, 0.9982436, 0.9952906, 0.9982355, -0.0022137, 0.0022841
6: 0.0036533, 0.0062893, 0.0036089, 0.0062821, -0.0020094, 0.0020733
7: -0.0097481, 0.0000891, -0.0099137, 0.0000621, -0.0074988, 0.0077371
8: -0.0092622, -0.0016059, -0.0092412, -0.0014770, -0.0060218, 0.0058363
9: -0.0038712, -0.0032106, -0.0038823, -0.0032124, -0.0005035, 0.0005195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015695
time: 1.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015741
time: 1.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0105700, -0.0036568, -0.0105185, -0.0036925, -0.0053230, 0.0053083
1: -0.0059187, -0.0039696, -0.0059042, -0.0039797, -0.0015007, 0.0014966
2: -0.0051098, 0.0092710, -0.0050028, 0.0091967, -0.0110728, 0.0110424
3: 0.0009511, 0.0028542, 0.0009652, 0.0028443, -0.0014653, 0.0014613
4: -0.0008367, 0.0099106, -0.0007812, 0.0098307, -0.0082524, 0.0082752
5: 0.9952738, 0.9982597, 0.9952892, 0.9982375, -0.0022928, 0.0022991
6: 0.0035937, 0.0063040, 0.0036077, 0.0062838, -0.0020811, 0.0020869
7: -0.0099706, 0.0001439, -0.0099183, 0.0000686, -0.0077664, 0.0077878
8: -0.0093048, -0.0014328, -0.0092463, -0.0014734, -0.0060613, 0.0060446
9: -0.0038861, -0.0032070, -0.0038826, -0.0032120, -0.0005215, 0.0005229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015796
time: 1.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015841
time: 1.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0104393, -0.0037317, -0.0104812, -0.0037705, -0.0052114, 0.0052725
1: -0.0058819, -0.0039908, -0.0058937, -0.0040017, -0.0014693, 0.0014865
2: -0.0048381, 0.0091152, -0.0049253, 0.0090345, -0.0108408, 0.0109678
3: 0.0009870, 0.0028335, 0.0009755, 0.0028229, -0.0014346, 0.0014514
4: -0.0007203, 0.0097076, -0.0006600, 0.0097727, -0.0081967, 0.0081017
5: 0.9953061, 0.9982033, 0.9953228, 0.9982213, -0.0022773, 0.0022509
6: 0.0036230, 0.0062528, 0.0036382, 0.0062692, -0.0020671, 0.0020431
7: -0.0098610, -0.0000472, -0.0098042, 0.0000141, -0.0077140, 0.0076246
8: -0.0091561, -0.0015180, -0.0092038, -0.0015622, -0.0059342, 0.0060038
9: -0.0038788, -0.0032198, -0.0038750, -0.0032157, -0.0005180, 0.0005120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015946, upper bound: 0.0015723
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015946, upper bound: 0.0015771
time: 1.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0104737, -0.0035859, -0.0104850, -0.0037673, -0.0052445, 0.0054365
1: -0.0058916, -0.0039497, -0.0058948, -0.0040008, -0.0014786, 0.0015328
2: -0.0049096, 0.0094184, -0.0049330, 0.0090412, -0.0109096, 0.0113091
3: 0.0009776, 0.0028737, 0.0009745, 0.0028237, -0.0014437, 0.0014966
4: -0.0009468, 0.0097610, -0.0006649, 0.0097785, -0.0084517, 0.0081531
5: 0.9952433, 0.9982181, 0.9953215, 0.9982231, -0.0023481, 0.0022652
6: 0.0035659, 0.0062662, 0.0036370, 0.0062707, -0.0021314, 0.0020561
7: -0.0100742, 0.0000030, -0.0098089, 0.0000195, -0.0079540, 0.0076730
8: -0.0091952, -0.0013521, -0.0092081, -0.0015586, -0.0059719, 0.0061906
9: -0.0038931, -0.0032164, -0.0038753, -0.0032153, -0.0005341, 0.0005152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015946, upper bound: 0.0015819
time: 2.18 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015946, upper bound: 0.0015862
time: 2.13 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0104393, -0.0037317, -0.0105141, -0.0036957, -0.0051258, 0.0051554
1: -0.0058819, -0.0039908, -0.0059030, -0.0039806, -0.0014451, 0.0014535
2: -0.0048381, 0.0091152, -0.0049936, 0.0091901, -0.0106626, 0.0107242
3: 0.0009870, 0.0028335, 0.0009665, 0.0028435, -0.0014110, 0.0014192
4: -0.0007203, 0.0097076, -0.0007763, 0.0098238, -0.0080146, 0.0079686
5: 0.9953061, 0.9982033, 0.9952906, 0.9982355, -0.0022267, 0.0022139
6: 0.0036230, 0.0062528, 0.0036089, 0.0062821, -0.0020212, 0.0020096
7: -0.0098610, -0.0000472, -0.0099137, 0.0000621, -0.0075427, 0.0074993
8: -0.0091561, -0.0015180, -0.0092412, -0.0014770, -0.0058367, 0.0058705
9: -0.0038788, -0.0032198, -0.0038823, -0.0032124, -0.0005065, 0.0005036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015728
time: 1.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015775
time: 1.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0104737, -0.0035859, -0.0105185, -0.0036925, -0.0051575, 0.0053302
1: -0.0058916, -0.0039497, -0.0059042, -0.0039797, -0.0014541, 0.0015028
2: -0.0049096, 0.0094184, -0.0050028, 0.0091967, -0.0107287, 0.0110879
3: 0.0009776, 0.0028737, 0.0009652, 0.0028443, -0.0014198, 0.0014673
4: -0.0009468, 0.0097610, -0.0007812, 0.0098307, -0.0082864, 0.0080180
5: 0.9952433, 0.9982181, 0.9952892, 0.9982375, -0.0023022, 0.0022276
6: 0.0035659, 0.0062662, 0.0036077, 0.0062838, -0.0020897, 0.0020220
7: -0.0100742, 0.0000030, -0.0099183, 0.0000686, -0.0077984, 0.0075458
8: -0.0091952, -0.0013521, -0.0092463, -0.0014734, -0.0058729, 0.0060695
9: -0.0038931, -0.0032164, -0.0038826, -0.0032120, -0.0005237, 0.0005067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015947, upper bound: 0.0015829
time: 1.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015871
time: 1.96 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.07 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015421, upper bound: 0.0015412
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015475, upper bound: 0.0015418
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015420, upper bound: 0.0015528
IS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015475, upper bound: 0.0015528
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015420, upper bound: 0.0015486
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015474, upper bound: 0.0015486
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015423, upper bound: 0.0015595
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015474, upper bound: 0.0015598
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015493, upper bound: 0.0015415
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015513, upper bound: 0.0015418
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015494, upper bound: 0.0015522
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015514, upper bound: 0.0015528
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015494, upper bound: 0.0015483
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015514, upper bound: 0.0015486
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015496, upper bound: 0.0015593
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015513, upper bound: 0.0015595
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015741, upper bound: 0.0015835
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015851, upper bound: 0.0015835
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015761, upper bound: 0.0015832
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015849, upper bound: 0.0015838
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015741, upper bound: 0.0015917
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015851, upper bound: 0.0015918
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015758, upper bound: 0.0015915
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015849, upper bound: 0.0015915
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015723, upper bound: 0.0015861
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015822, upper bound: 0.0015863
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015776, upper bound: 0.0015860
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015868, upper bound: 0.0015863
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015723, upper bound: 0.0015937
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015822, upper bound: 0.0015934
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015776, upper bound: 0.0015937
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015868, upper bound: 0.0015939
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015497, upper bound: 0.0015412
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015566, upper bound: 0.0015415
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015564, upper bound: 0.0015418
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015597, upper bound: 0.0015415
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015495, upper bound: 0.0015414
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015566, upper bound: 0.0015417
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015565, upper bound: 0.0015420
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015598, upper bound: 0.0015417
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015494, upper bound: 0.0015528
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015566, upper bound: 0.0015523
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015563, upper bound: 0.0015528
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015597, upper bound: 0.0015526
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015495, upper bound: 0.0015530
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015567, upper bound: 0.0015527
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015565, upper bound: 0.0015528
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015598, upper bound: 0.0015525
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015690
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015742
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015917, upper bound: 0.0015791
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015920, upper bound: 0.0015831
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015695
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015741
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015796
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015918, upper bound: 0.0015841
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015946, upper bound: 0.0015723
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015946, upper bound: 0.0015771
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015946, upper bound: 0.0015819
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015946, upper bound: 0.0015862
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015728
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015775
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015947, upper bound: 0.0015829
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.07
Output dim: 5, lower bound: -0.0015948, upper bound: 0.0015871

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0105498, -0.0039394, -0.0104036, -0.0038219, -0.0052464, 0.0049590
1: -0.0059131, -0.0040493, -0.0058718, -0.0040162, -0.0014792, 0.0013981
2: -0.0050680, 0.0086832, -0.0047638, 0.0089276, -0.0109136, 0.0103157
3: 0.0009566, 0.0027764, 0.0009969, 0.0028087, -0.0014442, 0.0013651
4: -0.0003974, 0.0098794, -0.0005801, 0.0096520, -0.0077093, 0.0081562
5: 0.9953958, 0.9982510, 0.9953451, 0.9981879, -0.0021419, 0.0022660
6: 0.0037044, 0.0062961, 0.0036584, 0.0062388, -0.0019442, 0.0020569
7: -0.0095571, 0.0001144, -0.0097290, -0.0000995, -0.0072553, 0.0076759
8: -0.0092819, -0.0017545, -0.0091154, -0.0016207, -0.0059741, 0.0056468
9: -0.0038584, -0.0032089, -0.0038699, -0.0032233, -0.0004872, 0.0005154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015402, upper bound: 0.0015493
time: 1.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015403, upper bound: 0.0015514
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0105548, -0.0039363, -0.0104366, -0.0036797, -0.0054099, 0.0049916
1: -0.0059145, -0.0040485, -0.0058811, -0.0039761, -0.0015253, 0.0014073
2: -0.0050783, 0.0086895, -0.0048323, 0.0092234, -0.0112537, 0.0103836
3: 0.0009553, 0.0027772, 0.0009878, 0.0028479, -0.0014892, 0.0013741
4: -0.0004021, 0.0098871, -0.0008011, 0.0097032, -0.0077600, 0.0084103
5: 0.9953945, 0.9982532, 0.9952837, 0.9982021, -0.0021560, 0.0023366
6: 0.0037033, 0.0062980, 0.0036026, 0.0062517, -0.0019570, 0.0021210
7: -0.0095616, 0.0001217, -0.0099371, -0.0000513, -0.0073031, 0.0079150
8: -0.0092876, -0.0017511, -0.0091529, -0.0014588, -0.0061603, 0.0056840
9: -0.0038587, -0.0032084, -0.0038839, -0.0032201, -0.0004904, 0.0005315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015534, upper bound: 0.0015496
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015534, upper bound: 0.0015514
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0107102, -0.0039276, -0.0103941, -0.0038634, -0.0054595, 0.0050069
1: -0.0059583, -0.0040460, -0.0058691, -0.0040279, -0.0015392, 0.0014116
2: -0.0054015, 0.0087077, -0.0047440, 0.0088412, -0.0113570, 0.0104155
3: 0.0009125, 0.0027796, 0.0009995, 0.0027973, -0.0015029, 0.0013783
4: -0.0004157, 0.0101286, -0.0005155, 0.0096372, -0.0077839, 0.0084875
5: 0.9953907, 0.9983202, 0.9953630, 0.9981838, -0.0021626, 0.0023581
6: 0.0036998, 0.0063590, 0.0036747, 0.0062350, -0.0019630, 0.0021404
7: -0.0095744, 0.0003490, -0.0096683, -0.0001134, -0.0073255, 0.0079877
8: -0.0094645, -0.0017411, -0.0091046, -0.0016680, -0.0062168, 0.0057014
9: -0.0038595, -0.0031932, -0.0038658, -0.0032242, -0.0004919, 0.0005364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015415, upper bound: 0.0015493
time: 2.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015415, upper bound: 0.0015514
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0107141, -0.0039246, -0.0104269, -0.0037222, -0.0056141, 0.0050390
1: -0.0059594, -0.0040452, -0.0058784, -0.0039881, -0.0015828, 0.0014207
2: -0.0054096, 0.0087139, -0.0048123, 0.0091350, -0.0116784, 0.0104821
3: 0.0009114, 0.0027804, 0.0009905, 0.0028362, -0.0015454, 0.0013871
4: -0.0004203, 0.0101346, -0.0007350, 0.0096883, -0.0078336, 0.0087277
5: 0.9953895, 0.9983220, 0.9953021, 0.9981979, -0.0021764, 0.0024248
6: 0.0036987, 0.0063605, 0.0036193, 0.0062479, -0.0019755, 0.0022010
7: -0.0095787, 0.0003547, -0.0098749, -0.0000654, -0.0073723, 0.0082137
8: -0.0094689, -0.0017377, -0.0091420, -0.0015072, -0.0063928, 0.0057379
9: -0.0038598, -0.0031928, -0.0038797, -0.0032210, -0.0004950, 0.0005515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015526, upper bound: 0.0015493
time: 1.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015526, upper bound: 0.0015516
time: 1.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0105498, -0.0039394, -0.0104360, -0.0037462, -0.0053909, 0.0050703
1: -0.0059131, -0.0040493, -0.0058809, -0.0039949, -0.0015199, 0.0014295
2: -0.0050680, 0.0086832, -0.0048311, 0.0090850, -0.0112141, 0.0105472
3: 0.0009566, 0.0027764, 0.0009880, 0.0028295, -0.0014840, 0.0013958
4: -0.0003974, 0.0098794, -0.0006977, 0.0097023, -0.0078823, 0.0083807
5: 0.9953958, 0.9982510, 0.9953124, 0.9982018, -0.0021899, 0.0023284
6: 0.0037044, 0.0062961, 0.0036287, 0.0062515, -0.0019878, 0.0021135
7: -0.0095571, 0.0001144, -0.0098397, -0.0000522, -0.0074181, 0.0078872
8: -0.0092819, -0.0017545, -0.0091523, -0.0015346, -0.0061386, 0.0057735
9: -0.0038584, -0.0032089, -0.0038773, -0.0032201, -0.0004981, 0.0005296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015402, upper bound: 0.0015561
time: 2.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015403, upper bound: 0.0015597
time: 1.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0105548, -0.0039363, -0.0104703, -0.0035996, -0.0055385, 0.0051015
1: -0.0059145, -0.0040485, -0.0058906, -0.0039535, -0.0015615, 0.0014383
2: -0.0050783, 0.0086895, -0.0049026, 0.0093900, -0.0115211, 0.0106121
3: 0.0009553, 0.0027772, 0.0009785, 0.0028699, -0.0015246, 0.0014043
4: -0.0004021, 0.0098871, -0.0009256, 0.0097557, -0.0079308, 0.0086102
5: 0.9953945, 0.9982532, 0.9952490, 0.9982167, -0.0022034, 0.0023922
6: 0.0037033, 0.0062980, 0.0035712, 0.0062649, -0.0020000, 0.0021714
7: -0.0095616, 0.0001217, -0.0100543, -0.0000019, -0.0074638, 0.0081031
8: -0.0092876, -0.0017511, -0.0091914, -0.0013676, -0.0063067, 0.0058091
9: -0.0038587, -0.0032084, -0.0038917, -0.0032167, -0.0005012, 0.0005441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015534, upper bound: 0.0015567
time: 2.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015534, upper bound: 0.0015597
time: 1.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0107102, -0.0039276, -0.0104267, -0.0037882, -0.0055987, 0.0051188
1: -0.0059583, -0.0040460, -0.0058783, -0.0040067, -0.0015785, 0.0014432
2: -0.0054015, 0.0087077, -0.0048117, 0.0089977, -0.0116465, 0.0106481
3: 0.0009125, 0.0027796, 0.0009905, 0.0028180, -0.0015412, 0.0014091
4: -0.0004157, 0.0101286, -0.0006325, 0.0096879, -0.0079577, 0.0087039
5: 0.9953907, 0.9983202, 0.9953305, 0.9981979, -0.0022109, 0.0024182
6: 0.0036998, 0.0063590, 0.0036452, 0.0062478, -0.0020068, 0.0021950
7: -0.0095744, 0.0003490, -0.0097783, -0.0000658, -0.0074891, 0.0081913
8: -0.0094645, -0.0017411, -0.0091417, -0.0015824, -0.0063753, 0.0058288
9: -0.0038595, -0.0031932, -0.0038732, -0.0032210, -0.0005029, 0.0005500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015418, upper bound: 0.0015564
time: 1.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015418, upper bound: 0.0015596
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0107141, -0.0039246, -0.0104608, -0.0036458, -0.0057400, 0.0051492
1: -0.0059594, -0.0040452, -0.0058880, -0.0039666, -0.0016183, 0.0014518
2: -0.0054096, 0.0087139, -0.0048828, 0.0092938, -0.0119403, 0.0107115
3: 0.0009114, 0.0027804, 0.0009811, 0.0028572, -0.0015801, 0.0014175
4: -0.0004203, 0.0101346, -0.0008537, 0.0097409, -0.0080051, 0.0089234
5: 0.9953895, 0.9983220, 0.9952691, 0.9982126, -0.0022241, 0.0024792
6: 0.0036987, 0.0063605, 0.0035894, 0.0062612, -0.0020188, 0.0022504
7: -0.0095787, 0.0003547, -0.0099866, -0.0000158, -0.0075337, 0.0083979
8: -0.0094689, -0.0017377, -0.0091805, -0.0014203, -0.0065361, 0.0058635
9: -0.0038598, -0.0031928, -0.0038872, -0.0032177, -0.0005059, 0.0005639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015526, upper bound: 0.0015564
time: 1.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015526, upper bound: 0.0015597
time: 1.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0104564, -0.0038748, -0.0104036, -0.0038219, -0.0050804, 0.0049706
1: -0.0058867, -0.0040311, -0.0058718, -0.0040162, -0.0014324, 0.0014014
2: -0.0048735, 0.0088175, -0.0047638, 0.0089276, -0.0105684, 0.0103399
3: 0.0009824, 0.0027941, 0.0009969, 0.0028087, -0.0013986, 0.0013683
4: -0.0004977, 0.0097340, -0.0005801, 0.0096520, -0.0077274, 0.0078981
5: 0.9953679, 0.9982107, 0.9953451, 0.9981879, -0.0021469, 0.0021943
6: 0.0036791, 0.0062595, 0.0036584, 0.0062388, -0.0019487, 0.0019918
7: -0.0096516, -0.0000223, -0.0097290, -0.0000995, -0.0072724, 0.0074330
8: -0.0091755, -0.0016810, -0.0091154, -0.0016207, -0.0057851, 0.0056601
9: -0.0038647, -0.0032181, -0.0038699, -0.0032233, -0.0004883, 0.0004991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015404, upper bound: 0.0015530
time: 2.06 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015404, upper bound: 0.0015552
time: 2.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0104601, -0.0038717, -0.0104366, -0.0036797, -0.0052580, 0.0050026
1: -0.0058878, -0.0040302, -0.0058811, -0.0039761, -0.0014824, 0.0014104
2: -0.0048813, 0.0088239, -0.0048323, 0.0092234, -0.0109377, 0.0104064
3: 0.0009813, 0.0027950, 0.0009878, 0.0028479, -0.0014474, 0.0013771
4: -0.0005026, 0.0097399, -0.0008011, 0.0097032, -0.0077771, 0.0081742
5: 0.9953666, 0.9982122, 0.9952837, 0.9982021, -0.0021607, 0.0022710
6: 0.0036779, 0.0062609, 0.0036026, 0.0062517, -0.0019613, 0.0020614
7: -0.0096561, -0.0000168, -0.0099371, -0.0000513, -0.0073191, 0.0076928
8: -0.0091798, -0.0016775, -0.0091529, -0.0014588, -0.0059873, 0.0056965
9: -0.0038650, -0.0032178, -0.0038839, -0.0032201, -0.0004915, 0.0005166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015520, upper bound: 0.0015533
time: 1.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015521, upper bound: 0.0015555
time: 2.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106379, -0.0038413, -0.0103941, -0.0038634, -0.0053206, 0.0050395
1: -0.0059379, -0.0040217, -0.0058691, -0.0040279, -0.0015001, 0.0014208
2: -0.0052511, 0.0088872, -0.0047440, 0.0088412, -0.0110680, 0.0104831
3: 0.0009324, 0.0028034, 0.0009995, 0.0027973, -0.0014647, 0.0013873
4: -0.0005499, 0.0100162, -0.0005155, 0.0096372, -0.0078344, 0.0082715
5: 0.9953534, 0.9982890, 0.9953630, 0.9981838, -0.0021766, 0.0022981
6: 0.0036660, 0.0063306, 0.0036747, 0.0062350, -0.0019757, 0.0020860
7: -0.0097006, 0.0002432, -0.0096683, -0.0001134, -0.0073731, 0.0077844
8: -0.0093822, -0.0016429, -0.0091046, -0.0016680, -0.0060586, 0.0057385
9: -0.0038680, -0.0032003, -0.0038658, -0.0032242, -0.0004951, 0.0005227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015453, upper bound: 0.0015531
time: 2.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015455, upper bound: 0.0015552
time: 2.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106428, -0.0038382, -0.0104269, -0.0037222, -0.0054821, 0.0050709
1: -0.0059393, -0.0040208, -0.0058784, -0.0039881, -0.0015456, 0.0014297
2: -0.0052614, 0.0088936, -0.0048123, 0.0091350, -0.0114039, 0.0105485
3: 0.0009310, 0.0028042, 0.0009905, 0.0028362, -0.0015091, 0.0013959
4: -0.0005546, 0.0100239, -0.0007350, 0.0096883, -0.0078833, 0.0085226
5: 0.9953522, 0.9982913, 0.9953021, 0.9981979, -0.0021902, 0.0023678
6: 0.0036648, 0.0063325, 0.0036193, 0.0062479, -0.0019881, 0.0021493
7: -0.0097051, 0.0002505, -0.0098749, -0.0000654, -0.0074191, 0.0080207
8: -0.0093878, -0.0016394, -0.0091420, -0.0015072, -0.0062425, 0.0057743
9: -0.0038683, -0.0031998, -0.0038797, -0.0032210, -0.0004982, 0.0005386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015533
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015557
time: 2.20 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0104564, -0.0038748, -0.0104360, -0.0037462, -0.0052308, 0.0050880
1: -0.0058867, -0.0040311, -0.0058809, -0.0039949, -0.0014748, 0.0014345
2: -0.0048735, 0.0088175, -0.0048311, 0.0090850, -0.0108812, 0.0105842
3: 0.0009824, 0.0027941, 0.0009880, 0.0028295, -0.0014400, 0.0014006
4: -0.0004977, 0.0097340, -0.0006977, 0.0097023, -0.0079099, 0.0081319
5: 0.9953679, 0.9982107, 0.9953124, 0.9982018, -0.0021976, 0.0022593
6: 0.0036791, 0.0062595, 0.0036287, 0.0062515, -0.0019948, 0.0020508
7: -0.0096516, -0.0000223, -0.0098397, -0.0000522, -0.0074441, 0.0076531
8: -0.0091755, -0.0016810, -0.0091523, -0.0015346, -0.0059564, 0.0057938
9: -0.0038647, -0.0032181, -0.0038773, -0.0032201, -0.0004999, 0.0005139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015402, upper bound: 0.0015594
time: 1.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015404, upper bound: 0.0015625
time: 2.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0104601, -0.0038717, -0.0104703, -0.0035996, -0.0053959, 0.0051212
1: -0.0058878, -0.0040302, -0.0058906, -0.0039535, -0.0015213, 0.0014439
2: -0.0048813, 0.0088239, -0.0049026, 0.0093900, -0.0112247, 0.0106531
3: 0.0009813, 0.0027950, 0.0009785, 0.0028699, -0.0014854, 0.0014098
4: -0.0005026, 0.0097399, -0.0009256, 0.0097557, -0.0079615, 0.0083886
5: 0.9953666, 0.9982122, 0.9952490, 0.9982167, -0.0022119, 0.0023306
6: 0.0036779, 0.0062609, 0.0035712, 0.0062649, -0.0020078, 0.0021155
7: -0.0096561, -0.0000168, -0.0100543, -0.0000019, -0.0074926, 0.0078946
8: -0.0091798, -0.0016775, -0.0091914, -0.0013676, -0.0061444, 0.0058315
9: -0.0038650, -0.0032178, -0.0038917, -0.0032167, -0.0005031, 0.0005301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015520, upper bound: 0.0015597
time: 2.06 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015521, upper bound: 0.0015630
time: 1.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106379, -0.0038413, -0.0104267, -0.0037882, -0.0054671, 0.0051573
1: -0.0059379, -0.0040217, -0.0058783, -0.0040067, -0.0015414, 0.0014540
2: -0.0052511, 0.0088872, -0.0048117, 0.0089977, -0.0113727, 0.0107282
3: 0.0009324, 0.0028034, 0.0009905, 0.0028180, -0.0015050, 0.0014197
4: -0.0005499, 0.0100162, -0.0006325, 0.0096879, -0.0080176, 0.0084992
5: 0.9953534, 0.9982890, 0.9953305, 0.9981979, -0.0022275, 0.0023613
6: 0.0036660, 0.0063306, 0.0036452, 0.0062478, -0.0020219, 0.0021434
7: -0.0097006, 0.0002432, -0.0097783, -0.0000658, -0.0075455, 0.0079987
8: -0.0093822, -0.0016429, -0.0091417, -0.0015824, -0.0062254, 0.0058726
9: -0.0038680, -0.0032003, -0.0038732, -0.0032210, -0.0005067, 0.0005371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015455, upper bound: 0.0015592
time: 2.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015452, upper bound: 0.0015628
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106428, -0.0038382, -0.0104608, -0.0036458, -0.0056188, 0.0051899
1: -0.0059393, -0.0040208, -0.0058880, -0.0039666, -0.0015841, 0.0014632
2: -0.0052614, 0.0088936, -0.0048828, 0.0092938, -0.0116882, 0.0107960
3: 0.0009310, 0.0028042, 0.0009811, 0.0028572, -0.0015468, 0.0014287
4: -0.0005546, 0.0100239, -0.0008537, 0.0097409, -0.0080682, 0.0087351
5: 0.9953522, 0.9982913, 0.9952691, 0.9982126, -0.0022416, 0.0024269
6: 0.0036648, 0.0063325, 0.0035894, 0.0062612, -0.0020347, 0.0022029
7: -0.0097051, 0.0002505, -0.0099866, -0.0000158, -0.0075931, 0.0082207
8: -0.0093878, -0.0016394, -0.0091805, -0.0014203, -0.0063982, 0.0059097
9: -0.0038683, -0.0031998, -0.0038872, -0.0032177, -0.0005099, 0.0005520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015592
time: 2.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015628
time: 2.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0105292, -0.0038224, -0.0104564, -0.0038748, -0.0052473, 0.0052228
1: -0.0059072, -0.0040163, -0.0058867, -0.0040311, -0.0014794, 0.0014725
2: -0.0050250, 0.0089264, -0.0048735, 0.0088175, -0.0109155, 0.0108646
3: 0.0009623, 0.0028086, 0.0009824, 0.0027941, -0.0014445, 0.0014378
4: -0.0005792, 0.0098472, -0.0004977, 0.0097340, -0.0081195, 0.0081576
5: 0.9953453, 0.9982421, 0.9953679, 0.9982107, -0.0022558, 0.0022664
6: 0.0036586, 0.0062880, 0.0036791, 0.0062595, -0.0020476, 0.0020572
7: -0.0097282, 0.0000842, -0.0096516, -0.0000223, -0.0076414, 0.0076772
8: -0.0092584, -0.0016214, -0.0091755, -0.0016810, -0.0059752, 0.0059473
9: -0.0038699, -0.0032110, -0.0038647, -0.0032181, -0.0005131, 0.0005155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015353
time: 2.17 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015355
time: 1.97 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0105194, -0.0038680, -0.0106379, -0.0038413, -0.0053115, 0.0054628
1: -0.0059045, -0.0040292, -0.0059379, -0.0040217, -0.0014975, 0.0015402
2: -0.0050046, 0.0088316, -0.0052511, 0.0088872, -0.0110490, 0.0113637
3: 0.0009650, 0.0027960, 0.0009324, 0.0028034, -0.0014622, 0.0015038
4: -0.0005083, 0.0098320, -0.0005499, 0.0100162, -0.0084925, 0.0082573
5: 0.9953650, 0.9982379, 0.9953534, 0.9982890, -0.0023595, 0.0022941
6: 0.0036765, 0.0062841, 0.0036660, 0.0063306, -0.0021417, 0.0020824
7: -0.0096615, 0.0000698, -0.0097006, 0.0002432, -0.0079924, 0.0077710
8: -0.0092472, -0.0016733, -0.0093822, -0.0016429, -0.0060482, 0.0062205
9: -0.0038654, -0.0032119, -0.0038680, -0.0032003, -0.0005367, 0.0005218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015414
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015595, upper bound: 0.0015404
time: 1.99 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0105666, -0.0036695, -0.0104601, -0.0038717, -0.0052825, 0.0053939
1: -0.0059178, -0.0039732, -0.0058878, -0.0040302, -0.0014893, 0.0015207
2: -0.0051027, 0.0092445, -0.0048813, 0.0088239, -0.0109887, 0.0112204
3: 0.0009520, 0.0028507, 0.0009813, 0.0027950, -0.0014542, 0.0014848
4: -0.0008169, 0.0099053, -0.0005026, 0.0097399, -0.0083855, 0.0082122
5: 0.9952793, 0.9982582, 0.9953666, 0.9982122, -0.0023297, 0.0022816
6: 0.0035987, 0.0063027, 0.0036779, 0.0062609, -0.0021147, 0.0020710
7: -0.0099519, 0.0001389, -0.0096561, -0.0000168, -0.0078916, 0.0077286
8: -0.0093010, -0.0014473, -0.0091798, -0.0016775, -0.0060152, 0.0061421
9: -0.0038849, -0.0032073, -0.0038650, -0.0032178, -0.0005299, 0.0005190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015422
time: 1.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015474
time: 1.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0105566, -0.0037208, -0.0106428, -0.0038382, -0.0053460, 0.0056261
1: -0.0059150, -0.0039877, -0.0059393, -0.0040208, -0.0015072, 0.0015862
2: -0.0050821, 0.0091379, -0.0052614, 0.0088936, -0.0111208, 0.0117034
3: 0.0009548, 0.0028365, 0.0009310, 0.0028042, -0.0014717, 0.0015488
4: -0.0007372, 0.0098899, -0.0005546, 0.0100239, -0.0087464, 0.0083110
5: 0.9953014, 0.9982539, 0.9953522, 0.9982913, -0.0024300, 0.0023090
6: 0.0036188, 0.0062988, 0.0036648, 0.0063325, -0.0022057, 0.0020959
7: -0.0098769, 0.0001244, -0.0097051, 0.0002505, -0.0082314, 0.0078215
8: -0.0092897, -0.0015056, -0.0093878, -0.0016394, -0.0060875, 0.0064065
9: -0.0038798, -0.0032083, -0.0038683, -0.0031998, -0.0005527, 0.0005252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015493
time: 1.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015514
time: 1.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0105292, -0.0038224, -0.0104894, -0.0037991, -0.0051611, 0.0050838
1: -0.0059072, -0.0040163, -0.0058960, -0.0040098, -0.0014551, 0.0014333
2: -0.0050250, 0.0089264, -0.0049422, 0.0089750, -0.0107362, 0.0105754
3: 0.0009623, 0.0028086, 0.0009733, 0.0028150, -0.0014208, 0.0013995
4: -0.0005792, 0.0098472, -0.0006155, 0.0097853, -0.0079034, 0.0080235
5: 0.9953453, 0.9982421, 0.9953352, 0.9982249, -0.0021958, 0.0022292
6: 0.0036586, 0.0062880, 0.0036495, 0.0062724, -0.0019931, 0.0020234
7: -0.0097282, 0.0000842, -0.0097624, 0.0000260, -0.0074380, 0.0075510
8: -0.0092584, -0.0016214, -0.0092131, -0.0015948, -0.0058770, 0.0057890
9: -0.0038699, -0.0032110, -0.0038721, -0.0032149, -0.0004994, 0.0005070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015595, upper bound: 0.0015352
time: 1.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015595, upper bound: 0.0015356
time: 2.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0105194, -0.0038680, -0.0106697, -0.0037704, -0.0052238, 0.0053320
1: -0.0059045, -0.0040292, -0.0059468, -0.0040017, -0.0014728, 0.0015033
2: -0.0050046, 0.0088316, -0.0053173, 0.0090346, -0.0108666, 0.0110917
3: 0.0009650, 0.0027960, 0.0009236, 0.0028229, -0.0014380, 0.0014678
4: -0.0005083, 0.0098320, -0.0006600, 0.0100657, -0.0082892, 0.0081210
5: 0.9953650, 0.9982379, 0.9953229, 0.9983028, -0.0023030, 0.0022563
6: 0.0036765, 0.0062841, 0.0036382, 0.0063431, -0.0020904, 0.0020480
7: -0.0096615, 0.0000698, -0.0098043, 0.0002898, -0.0078011, 0.0076428
8: -0.0092472, -0.0016733, -0.0094184, -0.0015622, -0.0059484, 0.0060716
9: -0.0038654, -0.0032119, -0.0038750, -0.0031972, -0.0005238, 0.0005132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015420
time: 2.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015407
time: 1.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0105666, -0.0036695, -0.0104939, -0.0037960, -0.0051959, 0.0052682
1: -0.0059178, -0.0039732, -0.0058973, -0.0040089, -0.0014649, 0.0014853
2: -0.0051027, 0.0092445, -0.0049516, 0.0089814, -0.0108085, 0.0109589
3: 0.0009520, 0.0028507, 0.0009720, 0.0028158, -0.0014303, 0.0014502
4: -0.0008169, 0.0099053, -0.0006203, 0.0097924, -0.0081900, 0.0080776
5: 0.9952793, 0.9982582, 0.9953339, 0.9982268, -0.0022754, 0.0022442
6: 0.0035987, 0.0063027, 0.0036482, 0.0062742, -0.0020654, 0.0020370
7: -0.0099519, 0.0001389, -0.0097669, 0.0000326, -0.0077077, 0.0076019
8: -0.0093010, -0.0014473, -0.0092182, -0.0015913, -0.0059166, 0.0059989
9: -0.0038849, -0.0032073, -0.0038724, -0.0032144, -0.0005176, 0.0005105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015420
time: 1.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015479
time: 1.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0105566, -0.0037208, -0.0106726, -0.0037674, -0.0052577, 0.0055024
1: -0.0059150, -0.0039877, -0.0059477, -0.0040008, -0.0014823, 0.0015513
2: -0.0050821, 0.0091379, -0.0053233, 0.0090409, -0.0109371, 0.0114460
3: 0.0009548, 0.0028365, 0.0009228, 0.0028237, -0.0014474, 0.0015147
4: -0.0007372, 0.0098899, -0.0006647, 0.0100702, -0.0085540, 0.0081737
5: 0.9953014, 0.9982539, 0.9953215, 0.9983040, -0.0023766, 0.0022709
6: 0.0036188, 0.0062988, 0.0036370, 0.0063442, -0.0021572, 0.0020613
7: -0.0098769, 0.0001244, -0.0098087, 0.0002940, -0.0080503, 0.0076924
8: -0.0092897, -0.0015056, -0.0094217, -0.0015587, -0.0059870, 0.0062656
9: -0.0038798, -0.0032083, -0.0038753, -0.0031969, -0.0005406, 0.0005165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015495
time: 1.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015519
time: 1.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0104360, -0.0037462, -0.0104564, -0.0038748, -0.0050880, 0.0052308
1: -0.0058809, -0.0039949, -0.0058867, -0.0040311, -0.0014345, 0.0014748
2: -0.0048311, 0.0090850, -0.0048735, 0.0088175, -0.0105842, 0.0108812
3: 0.0009880, 0.0028295, 0.0009824, 0.0027941, -0.0014006, 0.0014400
4: -0.0006977, 0.0097023, -0.0004977, 0.0097340, -0.0081319, 0.0079099
5: 0.9953124, 0.9982018, 0.9953679, 0.9982107, -0.0022593, 0.0021976
6: 0.0036287, 0.0062515, 0.0036791, 0.0062595, -0.0020508, 0.0019948
7: -0.0098397, -0.0000522, -0.0096516, -0.0000223, -0.0076531, 0.0074441
8: -0.0091523, -0.0015346, -0.0091755, -0.0016810, -0.0057938, 0.0059564
9: -0.0038773, -0.0032201, -0.0038647, -0.0032181, -0.0005139, 0.0004999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015398
time: 1.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015401
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0104267, -0.0037882, -0.0106379, -0.0038413, -0.0051573, 0.0054671
1: -0.0058783, -0.0040067, -0.0059379, -0.0040217, -0.0014540, 0.0015414
2: -0.0048117, 0.0089977, -0.0052511, 0.0088872, -0.0107282, 0.0113727
3: 0.0009905, 0.0028180, 0.0009324, 0.0028034, -0.0014197, 0.0015050
4: -0.0006325, 0.0096879, -0.0005499, 0.0100162, -0.0084992, 0.0080176
5: 0.9953305, 0.9981979, 0.9953534, 0.9982890, -0.0023613, 0.0022275
6: 0.0036452, 0.0062478, 0.0036660, 0.0063306, -0.0021434, 0.0020219
7: -0.0097783, -0.0000658, -0.0097006, 0.0002432, -0.0079987, 0.0075455
8: -0.0091417, -0.0015824, -0.0093822, -0.0016429, -0.0058726, 0.0062254
9: -0.0038732, -0.0032210, -0.0038680, -0.0032003, -0.0005371, 0.0005067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015454
time: 1.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015444
time: 2.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0104703, -0.0035996, -0.0104601, -0.0038717, -0.0051212, 0.0053959
1: -0.0058906, -0.0039535, -0.0058878, -0.0040302, -0.0014439, 0.0015213
2: -0.0049026, 0.0093900, -0.0048813, 0.0088239, -0.0106531, 0.0112247
3: 0.0009785, 0.0028699, 0.0009813, 0.0027950, -0.0014098, 0.0014854
4: -0.0009256, 0.0097557, -0.0005026, 0.0097399, -0.0083886, 0.0079615
5: 0.9952490, 0.9982167, 0.9953666, 0.9982122, -0.0023306, 0.0022119
6: 0.0035712, 0.0062649, 0.0036779, 0.0062609, -0.0021155, 0.0020078
7: -0.0100543, -0.0000019, -0.0096561, -0.0000168, -0.0078946, 0.0074926
8: -0.0091914, -0.0013676, -0.0091798, -0.0016775, -0.0058315, 0.0061444
9: -0.0038917, -0.0032167, -0.0038650, -0.0032178, -0.0005301, 0.0005031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015464
time: 1.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015517
time: 1.97 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0104608, -0.0036458, -0.0106428, -0.0038382, -0.0051899, 0.0056188
1: -0.0058880, -0.0039666, -0.0059393, -0.0040208, -0.0014632, 0.0015841
2: -0.0048828, 0.0092938, -0.0052614, 0.0088936, -0.0107960, 0.0116882
3: 0.0009811, 0.0028572, 0.0009310, 0.0028042, -0.0014287, 0.0015468
4: -0.0008537, 0.0097409, -0.0005546, 0.0100239, -0.0087351, 0.0080682
5: 0.9952691, 0.9982126, 0.9953522, 0.9982913, -0.0024269, 0.0022416
6: 0.0035894, 0.0062612, 0.0036648, 0.0063325, -0.0022029, 0.0020347
7: -0.0099866, -0.0000158, -0.0097051, 0.0002505, -0.0082207, 0.0075931
8: -0.0091805, -0.0014203, -0.0093878, -0.0016394, -0.0059097, 0.0063982
9: -0.0038872, -0.0032177, -0.0038683, -0.0031998, -0.0005520, 0.0005099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015640, upper bound: 0.0015527
time: 2.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015555
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0104360, -0.0037462, -0.0104894, -0.0037991, -0.0050026, 0.0051128
1: -0.0058809, -0.0039949, -0.0058960, -0.0040098, -0.0014104, 0.0014415
2: -0.0048311, 0.0090850, -0.0049422, 0.0089750, -0.0104064, 0.0106357
3: 0.0009880, 0.0028295, 0.0009733, 0.0028150, -0.0013771, 0.0014075
4: -0.0006977, 0.0097023, -0.0006155, 0.0097853, -0.0079484, 0.0077771
5: 0.9953124, 0.9982018, 0.9953352, 0.9982249, -0.0022083, 0.0021607
6: 0.0036287, 0.0062515, 0.0036495, 0.0062724, -0.0020045, 0.0019613
7: -0.0098397, -0.0000522, -0.0097624, 0.0000260, -0.0074804, 0.0073191
8: -0.0091523, -0.0015346, -0.0092131, -0.0015948, -0.0056965, 0.0058220
9: -0.0038773, -0.0032201, -0.0038721, -0.0032149, -0.0005023, 0.0004915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015397
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015399
time: 2.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0104267, -0.0037882, -0.0106697, -0.0037704, -0.0050692, 0.0053519
1: -0.0058783, -0.0040067, -0.0059468, -0.0040017, -0.0014292, 0.0015089
2: -0.0048117, 0.0089977, -0.0053173, 0.0090346, -0.0105450, 0.0111330
3: 0.0009905, 0.0028180, 0.0009236, 0.0028229, -0.0013955, 0.0014733
4: -0.0006325, 0.0096879, -0.0006600, 0.0100657, -0.0083201, 0.0078807
5: 0.9953305, 0.9981979, 0.9953229, 0.9983028, -0.0023116, 0.0021895
6: 0.0036452, 0.0062478, 0.0036382, 0.0063431, -0.0020982, 0.0019874
7: -0.0097783, -0.0000658, -0.0098043, 0.0002898, -0.0078302, 0.0074166
8: -0.0091417, -0.0015824, -0.0094184, -0.0015622, -0.0057724, 0.0060942
9: -0.0038732, -0.0032210, -0.0038750, -0.0031972, -0.0005258, 0.0004980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015456
time: 2.10 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015444
time: 2.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0104703, -0.0035996, -0.0104939, -0.0037960, -0.0050345, 0.0052897
1: -0.0058906, -0.0039535, -0.0058973, -0.0040089, -0.0014194, 0.0014914
2: -0.0049026, 0.0093900, -0.0049516, 0.0089814, -0.0104727, 0.0110037
3: 0.0009785, 0.0028699, 0.0009720, 0.0028158, -0.0013859, 0.0014562
4: -0.0009256, 0.0097557, -0.0006203, 0.0097924, -0.0082235, 0.0078266
5: 0.9952490, 0.9982167, 0.9953339, 0.9982268, -0.0022847, 0.0021745
6: 0.0035712, 0.0062649, 0.0036482, 0.0062742, -0.0020738, 0.0019738
7: -0.0100543, -0.0000019, -0.0097669, 0.0000326, -0.0077392, 0.0073657
8: -0.0091914, -0.0013676, -0.0092182, -0.0015913, -0.0057328, 0.0060234
9: -0.0038917, -0.0032167, -0.0038724, -0.0032144, -0.0005197, 0.0004946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015467
time: 2.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015522
time: 1.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0104608, -0.0036458, -0.0106726, -0.0037674, -0.0051010, 0.0055134
1: -0.0058880, -0.0039666, -0.0059477, -0.0040008, -0.0014382, 0.0015544
2: -0.0048828, 0.0092938, -0.0053233, 0.0090409, -0.0106111, 0.0114691
3: 0.0009811, 0.0028572, 0.0009228, 0.0028237, -0.0014042, 0.0015177
4: -0.0008537, 0.0097409, -0.0006647, 0.0100702, -0.0085713, 0.0079300
5: 0.9952691, 0.9982126, 0.9953215, 0.9983040, -0.0023814, 0.0022032
6: 0.0035894, 0.0062612, 0.0036370, 0.0063442, -0.0021615, 0.0019998
7: -0.0099866, -0.0000158, -0.0098087, 0.0002940, -0.0080665, 0.0074631
8: -0.0091805, -0.0014203, -0.0094217, -0.0015587, -0.0058085, 0.0062782
9: -0.0038872, -0.0032177, -0.0038753, -0.0031969, -0.0005417, 0.0005011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015534
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015556
time: 1.50 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.42 seconds
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015402, upper bound: 0.0015493
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015403, upper bound: 0.0015514
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015534, upper bound: 0.0015496
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015534, upper bound: 0.0015514
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015415, upper bound: 0.0015493
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015415, upper bound: 0.0015514
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015526, upper bound: 0.0015493
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015526, upper bound: 0.0015516
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015402, upper bound: 0.0015561
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015403, upper bound: 0.0015597
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015534, upper bound: 0.0015567
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015534, upper bound: 0.0015597
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015418, upper bound: 0.0015564
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015418, upper bound: 0.0015596
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015526, upper bound: 0.0015564
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015526, upper bound: 0.0015597
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015404, upper bound: 0.0015530
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015404, upper bound: 0.0015552
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015520, upper bound: 0.0015533
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015521, upper bound: 0.0015555
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015453, upper bound: 0.0015531
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015455, upper bound: 0.0015552
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015533
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015557
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015402, upper bound: 0.0015594
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015404, upper bound: 0.0015625
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015520, upper bound: 0.0015597
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015521, upper bound: 0.0015630
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015455, upper bound: 0.0015592
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015452, upper bound: 0.0015628
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015592
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015562, upper bound: 0.0015628
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015353
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015355
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015414
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015595, upper bound: 0.0015404
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015422
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015474
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015493
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015514
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015595, upper bound: 0.0015352
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015595, upper bound: 0.0015356
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015420
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015407
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015420
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015479
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015495
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015596, upper bound: 0.0015519
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015398
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015401
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015454
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015444
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015464
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015517
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015640, upper bound: 0.0015527
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015555
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015397
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015399
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015456
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015444
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015467
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015522
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015637, upper bound: 0.0015534
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 5, lower bound: -0.0015636, upper bound: 0.0015556

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 4.29 + 491.16 = 495.45 seconds
