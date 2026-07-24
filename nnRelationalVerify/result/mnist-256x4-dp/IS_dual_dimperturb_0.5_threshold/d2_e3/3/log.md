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
Threshold: 0.00031548


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0089523, -0.0060322, -0.0089523, -0.0060322, -0.0019720, 0.0019720)
1: (-0.0054627, -0.0046394, -0.0054627, -0.0046394, -0.0005560, 0.0005560)
2: (-0.0017448, 0.0043297, -0.0017448, 0.0043297, -0.0041022, 0.0041022)
3: (0.0013964, 0.0022003, 0.0013964, 0.0022003, -0.0005429, 0.0005429)
4: (0.0028561, 0.0073958, 0.0028561, 0.0073958, -0.0030658, 0.0030658)
5: (0.9962998, 0.9975610, 0.9962998, 0.9975610, -0.0008518, 0.0008518)
6: (0.0045249, 0.0056698, 0.0045249, 0.0056698, -0.0007731, 0.0007731)
7: (-0.0064952, -0.0022228, -0.0064952, -0.0022228, -0.0028852, 0.0028852)
8: (-0.0074628, -0.0041376, -0.0074628, -0.0041376, -0.0022456, 0.0022456)
9: (-0.0036528, -0.0033659, -0.0036528, -0.0033659, -0.0001937, 0.0001937)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 1.69 = 3.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0005390, upper bound: 0.0005391

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005143, upper bound: 0.0005159
time: 0.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005159, upper bound: 0.0005159
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.91 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 5, lower bound: -0.0005143, upper bound: 0.0005159
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 5, lower bound: -0.0005159, upper bound: 0.0005159

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0089765, -0.0061603, -0.0089505, -0.0060736, -0.0018489, 0.0018029
1: -0.0054695, -0.0046755, -0.0054621, -0.0046510, -0.0005213, 0.0005083
2: -0.0017952, 0.0040632, -0.0017410, 0.0042435, -0.0038461, 0.0037503
3: 0.0013897, 0.0021650, 0.0013969, 0.0021889, -0.0005090, 0.0004963
4: 0.0030553, 0.0074335, 0.0029205, 0.0073930, -0.0028027, 0.0028743
5: 0.9963551, 0.9975715, 0.9963176, 0.9975603, -0.0007787, 0.0007986
6: 0.0045752, 0.0056793, 0.0045412, 0.0056691, -0.0007068, 0.0007249
7: -0.0063078, -0.0021874, -0.0064346, -0.0022255, -0.0026377, 0.0027051
8: -0.0074904, -0.0042835, -0.0074607, -0.0041848, -0.0021054, 0.0020529
9: -0.0036402, -0.0033635, -0.0036487, -0.0033661, -0.0001771, 0.0001816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004800, upper bound: 0.0004818
time: 0.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004892, upper bound: 0.0004908
time: 0.85 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0089500, -0.0061082, -0.0089518, -0.0060487, -0.0019614, 0.0017477
1: -0.0054620, -0.0046608, -0.0054625, -0.0046440, -0.0005530, 0.0004927
2: -0.0017400, 0.0041715, -0.0017437, 0.0042953, -0.0040801, 0.0036356
3: 0.0013970, 0.0021793, 0.0013965, 0.0021957, -0.0005399, 0.0004811
4: 0.0029743, 0.0073922, 0.0028818, 0.0073950, -0.0027170, 0.0030492
5: 0.9963326, 0.9975600, 0.9963069, 0.9975608, -0.0007549, 0.0008472
6: 0.0045547, 0.0056689, 0.0045314, 0.0056696, -0.0006852, 0.0007690
7: -0.0063840, -0.0022263, -0.0064710, -0.0022236, -0.0025570, 0.0028696
8: -0.0074602, -0.0042242, -0.0074622, -0.0041565, -0.0022334, 0.0019901
9: -0.0036453, -0.0033661, -0.0036511, -0.0033659, -0.0001717, 0.0001927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004826, upper bound: 0.0004819
time: 0.80 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004911, upper bound: 0.0004911
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 5, lower bound: -0.0004800, upper bound: 0.0004818
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 5, lower bound: -0.0004892, upper bound: 0.0004908
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 5, lower bound: -0.0004826, upper bound: 0.0004819
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 5, lower bound: -0.0004911, upper bound: 0.0004911

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089724, -0.0062110, -0.0090814, -0.0062356, -0.0016085, 0.0017458
1: -0.0054683, -0.0046898, -0.0054990, -0.0046967, -0.0004535, 0.0004922
2: -0.0017867, 0.0039577, -0.0020133, 0.0039065, -0.0033460, 0.0036316
3: 0.0013909, 0.0021510, 0.0013609, 0.0021443, -0.0004428, 0.0004806
4: 0.0031341, 0.0074271, 0.0031724, 0.0075964, -0.0027140, 0.0025006
5: 0.9963770, 0.9975697, 0.9963877, 0.9976168, -0.0007540, 0.0006947
6: 0.0045951, 0.0056777, 0.0046047, 0.0057204, -0.0006844, 0.0006306
7: -0.0062336, -0.0021934, -0.0061976, -0.0020340, -0.0025542, 0.0023533
8: -0.0074857, -0.0043413, -0.0076098, -0.0043693, -0.0018316, 0.0019879
9: -0.0036352, -0.0033639, -0.0036328, -0.0033532, -0.0001715, 0.0001580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004801, upper bound: 0.0004800
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004801, upper bound: 0.0004818
time: 0.90 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089765, -0.0061603, -0.0089467, -0.0061218, -0.0016270, 0.0017996
1: -0.0054695, -0.0046755, -0.0054611, -0.0046646, -0.0004587, 0.0005074
2: -0.0017952, 0.0040632, -0.0017332, 0.0041433, -0.0033845, 0.0037436
3: 0.0013897, 0.0021650, 0.0013979, 0.0021756, -0.0004479, 0.0004954
4: 0.0030553, 0.0074335, 0.0029954, 0.0073871, -0.0027977, 0.0025294
5: 0.9963551, 0.9975715, 0.9963385, 0.9975587, -0.0007773, 0.0007027
6: 0.0045752, 0.0056793, 0.0045601, 0.0056676, -0.0007056, 0.0006379
7: -0.0063078, -0.0021874, -0.0063641, -0.0022310, -0.0026330, 0.0023804
8: -0.0074904, -0.0042835, -0.0074565, -0.0042397, -0.0018527, 0.0020493
9: -0.0036402, -0.0033635, -0.0036440, -0.0033664, -0.0001768, 0.0001598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004795, upper bound: 0.0004825
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004795, upper bound: 0.0004909
time: 0.84 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0089460, -0.0061580, -0.0090827, -0.0062109, -0.0017209, 0.0016793
1: -0.0054609, -0.0046748, -0.0054994, -0.0046897, -0.0004852, 0.0004735
2: -0.0017316, 0.0040679, -0.0020160, 0.0039579, -0.0035799, 0.0034934
3: 0.0013981, 0.0021656, 0.0013605, 0.0021511, -0.0004737, 0.0004623
4: 0.0030518, 0.0073860, 0.0031340, 0.0075985, -0.0026107, 0.0026754
5: 0.9963542, 0.9975584, 0.9963769, 0.9976173, -0.0007253, 0.0007433
6: 0.0045743, 0.0056673, 0.0045950, 0.0057209, -0.0006584, 0.0006747
7: -0.0063111, -0.0022321, -0.0062337, -0.0020321, -0.0024570, 0.0025178
8: -0.0074556, -0.0042810, -0.0076113, -0.0043412, -0.0019596, 0.0019123
9: -0.0036404, -0.0033665, -0.0036352, -0.0033531, -0.0001650, 0.0001691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004825, upper bound: 0.0004800
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004825, upper bound: 0.0004803
time: 1.07 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089500, -0.0061082, -0.0089481, -0.0060963, -0.0017483, 0.0017442
1: -0.0054620, -0.0046608, -0.0054615, -0.0046574, -0.0004929, 0.0004918
2: -0.0017400, 0.0041715, -0.0017359, 0.0041964, -0.0036368, 0.0036284
3: 0.0013970, 0.0021793, 0.0013976, 0.0021826, -0.0004813, 0.0004802
4: 0.0029743, 0.0073922, 0.0029558, 0.0073892, -0.0027116, 0.0027179
5: 0.9963326, 0.9975600, 0.9963274, 0.9975592, -0.0007534, 0.0007551
6: 0.0045547, 0.0056689, 0.0045501, 0.0056681, -0.0006838, 0.0006854
7: -0.0063840, -0.0022263, -0.0064014, -0.0022291, -0.0025519, 0.0025578
8: -0.0074602, -0.0042242, -0.0074580, -0.0042106, -0.0019908, 0.0019862
9: -0.0036453, -0.0033661, -0.0036465, -0.0033663, -0.0001714, 0.0001718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004910, upper bound: 0.0004887
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004910, upper bound: 0.0004890
time: 1.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.36 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 5, lower bound: -0.0004801, upper bound: 0.0004800
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 5, lower bound: -0.0004801, upper bound: 0.0004818
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 5, lower bound: -0.0004795, upper bound: 0.0004825
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 5, lower bound: -0.0004795, upper bound: 0.0004909
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 5, lower bound: -0.0004825, upper bound: 0.0004800
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 5, lower bound: -0.0004825, upper bound: 0.0004803
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 5, lower bound: -0.0004910, upper bound: 0.0004887
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 5, lower bound: -0.0004910, upper bound: 0.0004890

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0089724, -0.0062110, -0.0091100, -0.0063197, -0.0015007, 0.0016730
1: -0.0054683, -0.0046898, -0.0055071, -0.0047204, -0.0004231, 0.0004717
2: -0.0017867, 0.0039577, -0.0020729, 0.0037315, -0.0031218, 0.0034803
3: 0.0013909, 0.0021510, 0.0013530, 0.0021211, -0.0004131, 0.0004606
4: 0.0031341, 0.0074271, 0.0033032, 0.0076410, -0.0026009, 0.0023331
5: 0.9963770, 0.9975697, 0.9964240, 0.9976291, -0.0007226, 0.0006482
6: 0.0045951, 0.0056777, 0.0046377, 0.0057316, -0.0006559, 0.0005884
7: -0.0062336, -0.0021934, -0.0060745, -0.0019921, -0.0024478, 0.0021957
8: -0.0074857, -0.0043413, -0.0076424, -0.0044651, -0.0017089, 0.0019051
9: -0.0036352, -0.0033639, -0.0036245, -0.0033504, -0.0001644, 0.0001474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004651, upper bound: 0.0004707
time: 1.04 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004705, upper bound: 0.0004707
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0089724, -0.0062110, -0.0090809, -0.0062690, -0.0016362, 0.0017453
1: -0.0054683, -0.0046898, -0.0054989, -0.0047061, -0.0004613, 0.0004921
2: -0.0017867, 0.0039577, -0.0020122, 0.0038371, -0.0034035, 0.0036306
3: 0.0013909, 0.0021510, 0.0013610, 0.0021351, -0.0004504, 0.0004805
4: 0.0031341, 0.0074271, 0.0032242, 0.0075956, -0.0027133, 0.0025436
5: 0.9963770, 0.9975697, 0.9964020, 0.9976165, -0.0007538, 0.0007067
6: 0.0045951, 0.0056777, 0.0046178, 0.0057202, -0.0006843, 0.0006415
7: -0.0062336, -0.0021934, -0.0061488, -0.0020348, -0.0025535, 0.0023938
8: -0.0074857, -0.0043413, -0.0076092, -0.0044073, -0.0018631, 0.0019874
9: -0.0036352, -0.0033639, -0.0036295, -0.0033533, -0.0001715, 0.0001607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004651, upper bound: 0.0004728
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004705, upper bound: 0.0004728
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0091100, -0.0063197, -0.0089467, -0.0061218, -0.0018464, 0.0015642
1: -0.0055071, -0.0047204, -0.0054611, -0.0046646, -0.0005206, 0.0004410
2: -0.0020729, 0.0037315, -0.0017332, 0.0041433, -0.0038408, 0.0032540
3: 0.0013530, 0.0021211, 0.0013979, 0.0021756, -0.0005083, 0.0004306
4: 0.0033032, 0.0076410, 0.0029954, 0.0073871, -0.0024318, 0.0028704
5: 0.9964240, 0.9976291, 0.9963385, 0.9975587, -0.0006756, 0.0007975
6: 0.0046377, 0.0057316, 0.0045601, 0.0056676, -0.0006133, 0.0007239
7: -0.0060745, -0.0019921, -0.0063641, -0.0022310, -0.0022886, 0.0027014
8: -0.0076424, -0.0044651, -0.0074565, -0.0042397, -0.0021025, 0.0017812
9: -0.0036245, -0.0033504, -0.0036440, -0.0033664, -0.0001537, 0.0001814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004798
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004825
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089727, -0.0062092, -0.0089467, -0.0061218, -0.0016240, 0.0015895
1: -0.0054684, -0.0046893, -0.0054611, -0.0046646, -0.0004579, 0.0004481
2: -0.0017872, 0.0039615, -0.0017332, 0.0041433, -0.0033783, 0.0033065
3: 0.0013908, 0.0021515, 0.0013979, 0.0021756, -0.0004471, 0.0004376
4: 0.0031313, 0.0074275, 0.0029954, 0.0073871, -0.0024711, 0.0025247
5: 0.9963762, 0.9975698, 0.9963385, 0.9975587, -0.0006865, 0.0007014
6: 0.0045943, 0.0056778, 0.0045601, 0.0056676, -0.0006232, 0.0006367
7: -0.0062362, -0.0021930, -0.0063641, -0.0022310, -0.0023256, 0.0023761
8: -0.0074860, -0.0043392, -0.0074565, -0.0042397, -0.0018493, 0.0018100
9: -0.0036354, -0.0033639, -0.0036440, -0.0033664, -0.0001562, 0.0001595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004884
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004906
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0089460, -0.0061580, -0.0091100, -0.0063197, -0.0015637, 0.0018069
1: -0.0054609, -0.0046748, -0.0055071, -0.0047204, -0.0004409, 0.0005094
2: -0.0017316, 0.0040679, -0.0020729, 0.0037315, -0.0032528, 0.0037586
3: 0.0013981, 0.0021656, 0.0013530, 0.0021211, -0.0004305, 0.0004974
4: 0.0030518, 0.0073860, 0.0033032, 0.0076410, -0.0028090, 0.0024310
5: 0.9963542, 0.9975584, 0.9964240, 0.9976291, -0.0007804, 0.0006754
6: 0.0045743, 0.0056673, 0.0046377, 0.0057316, -0.0007084, 0.0006131
7: -0.0063111, -0.0022321, -0.0060745, -0.0019921, -0.0026436, 0.0022878
8: -0.0074556, -0.0042810, -0.0076424, -0.0044651, -0.0017806, 0.0020575
9: -0.0036404, -0.0033665, -0.0036245, -0.0033504, -0.0001775, 0.0001536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004671, upper bound: 0.0004707
time: 0.94 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004732, upper bound: 0.0004707
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0089460, -0.0061580, -0.0090809, -0.0062690, -0.0015061, 0.0016775
1: -0.0054609, -0.0046748, -0.0054989, -0.0047061, -0.0004246, 0.0004730
2: -0.0017316, 0.0040679, -0.0020122, 0.0038371, -0.0031329, 0.0034896
3: 0.0013981, 0.0021656, 0.0013610, 0.0021351, -0.0004146, 0.0004618
4: 0.0030518, 0.0073860, 0.0032242, 0.0075956, -0.0026079, 0.0023413
5: 0.9963542, 0.9975584, 0.9964020, 0.9976165, -0.0007246, 0.0006505
6: 0.0045743, 0.0056673, 0.0046178, 0.0057202, -0.0006577, 0.0005905
7: -0.0063111, -0.0022321, -0.0061488, -0.0020348, -0.0024543, 0.0022035
8: -0.0074556, -0.0042810, -0.0076092, -0.0044073, -0.0017150, 0.0019102
9: -0.0036404, -0.0033665, -0.0036295, -0.0033533, -0.0001648, 0.0001480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004671, upper bound: 0.0004711
time: 1.17 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004732, upper bound: 0.0004711
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0089500, -0.0061082, -0.0089727, -0.0062092, -0.0015919, 0.0018684
1: -0.0054620, -0.0046608, -0.0054684, -0.0046893, -0.0004488, 0.0005268
2: -0.0017400, 0.0041715, -0.0017872, 0.0039615, -0.0033115, 0.0038866
3: 0.0013970, 0.0021793, 0.0013908, 0.0021515, -0.0004382, 0.0005143
4: 0.0029743, 0.0073922, 0.0031313, 0.0074275, -0.0029046, 0.0024748
5: 0.9963326, 0.9975600, 0.9963762, 0.9975698, -0.0008070, 0.0006876
6: 0.0045547, 0.0056689, 0.0045943, 0.0056778, -0.0007325, 0.0006241
7: -0.0063840, -0.0022263, -0.0062362, -0.0021930, -0.0027335, 0.0023291
8: -0.0074602, -0.0042242, -0.0074860, -0.0043392, -0.0018127, 0.0021275
9: -0.0036453, -0.0033661, -0.0036354, -0.0033639, -0.0001836, 0.0001564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004809, upper bound: 0.0004798
time: 0.89 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004809, upper bound: 0.0004883
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0089500, -0.0061082, -0.0089462, -0.0061560, -0.0015273, 0.0017422
1: -0.0054620, -0.0046608, -0.0054609, -0.0046743, -0.0004306, 0.0004912
2: -0.0017400, 0.0041715, -0.0017322, 0.0040721, -0.0031771, 0.0036241
3: 0.0013970, 0.0021793, 0.0013981, 0.0021662, -0.0004204, 0.0004796
4: 0.0029743, 0.0073922, 0.0030487, 0.0073864, -0.0027084, 0.0023744
5: 0.9963326, 0.9975600, 0.9963533, 0.9975584, -0.0007525, 0.0006597
6: 0.0045547, 0.0056689, 0.0045735, 0.0056674, -0.0006830, 0.0005988
7: -0.0063840, -0.0022263, -0.0063140, -0.0022317, -0.0025489, 0.0022346
8: -0.0074602, -0.0042242, -0.0074559, -0.0042787, -0.0017392, 0.0019838
9: -0.0036453, -0.0033661, -0.0036406, -0.0033665, -0.0001712, 0.0001500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004809, upper bound: 0.0004800
time: 1.18 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004809, upper bound: 0.0004886
time: 1.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.77 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004651, upper bound: 0.0004707
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004705, upper bound: 0.0004707
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004651, upper bound: 0.0004728
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004705, upper bound: 0.0004728
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004798
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004825
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004884
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004794, upper bound: 0.0004906
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004671, upper bound: 0.0004707
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004732, upper bound: 0.0004707
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004671, upper bound: 0.0004711
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004732, upper bound: 0.0004711
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004809, upper bound: 0.0004798
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004809, upper bound: 0.0004883
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004809, upper bound: 0.0004800
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 5, lower bound: -0.0004809, upper bound: 0.0004886

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0089586, -0.0062113, -0.0090544, -0.0063208, -0.0014858, 0.0016221
1: -0.0054644, -0.0046898, -0.0054914, -0.0047207, -0.0004189, 0.0004573
2: -0.0017579, 0.0039571, -0.0019572, 0.0037294, -0.0030908, 0.0033742
3: 0.0013947, 0.0021510, 0.0013683, 0.0021208, -0.0004090, 0.0004465
4: 0.0031346, 0.0074056, 0.0033048, 0.0075546, -0.0025217, 0.0023099
5: 0.9963771, 0.9975638, 0.9964244, 0.9976051, -0.0007006, 0.0006417
6: 0.0045952, 0.0056723, 0.0046381, 0.0057098, -0.0006359, 0.0005825
7: -0.0062332, -0.0022136, -0.0060730, -0.0020734, -0.0023732, 0.0021738
8: -0.0074700, -0.0043416, -0.0075791, -0.0044662, -0.0016919, 0.0018471
9: -0.0036352, -0.0033653, -0.0036244, -0.0033558, -0.0001594, 0.0001460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004048, upper bound: 0.0004064
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004019, upper bound: 0.0003966
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0089580, -0.0062114, -0.0090735, -0.0062855, -0.0015162, 0.0016337
1: -0.0054642, -0.0046899, -0.0054968, -0.0047108, -0.0004275, 0.0004606
2: -0.0017566, 0.0039568, -0.0019969, 0.0038027, -0.0031540, 0.0033984
3: 0.0013948, 0.0021509, 0.0013630, 0.0021305, -0.0004174, 0.0004497
4: 0.0031348, 0.0074046, 0.0032500, 0.0075842, -0.0025398, 0.0023571
5: 0.9963772, 0.9975635, 0.9964092, 0.9976133, -0.0007056, 0.0006549
6: 0.0045952, 0.0056720, 0.0046243, 0.0057173, -0.0006405, 0.0005944
7: -0.0062329, -0.0022146, -0.0061245, -0.0020455, -0.0023902, 0.0022183
8: -0.0074693, -0.0043418, -0.0076008, -0.0044261, -0.0017265, 0.0018603
9: -0.0036351, -0.0033653, -0.0036279, -0.0033540, -0.0001605, 0.0001490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004068, upper bound: 0.0004055
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004030, upper bound: 0.0003949
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0089586, -0.0062113, -0.0090264, -0.0062699, -0.0016212, 0.0016906
1: -0.0054644, -0.0046898, -0.0054835, -0.0047064, -0.0004571, 0.0004767
2: -0.0017579, 0.0039571, -0.0018990, 0.0038352, -0.0033725, 0.0035169
3: 0.0013947, 0.0021510, 0.0013760, 0.0021348, -0.0004463, 0.0004654
4: 0.0031346, 0.0074056, 0.0032257, 0.0075111, -0.0026283, 0.0025204
5: 0.9963771, 0.9975638, 0.9964024, 0.9975930, -0.0007302, 0.0007002
6: 0.0045952, 0.0056723, 0.0046181, 0.0056988, -0.0006628, 0.0006356
7: -0.0062332, -0.0022136, -0.0061474, -0.0021144, -0.0024735, 0.0023720
8: -0.0074700, -0.0043416, -0.0075472, -0.0044083, -0.0018461, 0.0019251
9: -0.0036352, -0.0033653, -0.0036294, -0.0033586, -0.0001661, 0.0001593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004048, upper bound: 0.0004102
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004017, upper bound: 0.0003986
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0089580, -0.0062114, -0.0090446, -0.0062360, -0.0016608, 0.0017031
1: -0.0054642, -0.0046899, -0.0054887, -0.0046968, -0.0004682, 0.0004802
2: -0.0017566, 0.0039568, -0.0019368, 0.0039057, -0.0034547, 0.0035427
3: 0.0013948, 0.0021509, 0.0013710, 0.0021442, -0.0004572, 0.0004688
4: 0.0031348, 0.0074046, 0.0031730, 0.0075393, -0.0026476, 0.0025818
5: 0.9963772, 0.9975635, 0.9963878, 0.9976009, -0.0007356, 0.0007173
6: 0.0045952, 0.0056720, 0.0046048, 0.0057060, -0.0006677, 0.0006511
7: -0.0062329, -0.0022146, -0.0061970, -0.0020878, -0.0024917, 0.0024298
8: -0.0074693, -0.0043418, -0.0075679, -0.0043697, -0.0018911, 0.0019393
9: -0.0036351, -0.0033653, -0.0036327, -0.0033568, -0.0001673, 0.0001632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004067, upper bound: 0.0004097
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004030, upper bound: 0.0003975
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0091100, -0.0063197, -0.0089727, -0.0062092, -0.0017368, 0.0015007
1: -0.0055071, -0.0047204, -0.0054684, -0.0046893, -0.0004897, 0.0004231
2: -0.0020729, 0.0037315, -0.0017872, 0.0039615, -0.0036128, 0.0031218
3: 0.0013530, 0.0021211, 0.0013908, 0.0021515, -0.0004781, 0.0004131
4: 0.0033032, 0.0076410, 0.0031313, 0.0074275, -0.0023330, 0.0027000
5: 0.9964240, 0.9976291, 0.9963762, 0.9975698, -0.0006482, 0.0007501
6: 0.0046377, 0.0057316, 0.0045943, 0.0056778, -0.0005884, 0.0006809
7: -0.0060745, -0.0019921, -0.0062362, -0.0021930, -0.0021956, 0.0025410
8: -0.0076424, -0.0044651, -0.0074860, -0.0043392, -0.0019777, 0.0017089
9: -0.0036245, -0.0033504, -0.0036354, -0.0033639, -0.0001474, 0.0001706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004715, upper bound: 0.0004648
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004714, upper bound: 0.0004701
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0091100, -0.0063197, -0.0089462, -0.0061560, -0.0018662, 0.0015637
1: -0.0055071, -0.0047204, -0.0054609, -0.0046743, -0.0005262, 0.0004409
2: -0.0020729, 0.0037315, -0.0017322, 0.0040721, -0.0038821, 0.0032529
3: 0.0013530, 0.0021211, 0.0013981, 0.0021662, -0.0005137, 0.0004305
4: 0.0033032, 0.0076410, 0.0030487, 0.0073864, -0.0024310, 0.0029013
5: 0.9964240, 0.9976291, 0.9963533, 0.9975584, -0.0006754, 0.0008061
6: 0.0046377, 0.0057316, 0.0045735, 0.0056674, -0.0006131, 0.0007317
7: -0.0060745, -0.0019921, -0.0063140, -0.0022317, -0.0022879, 0.0027304
8: -0.0076424, -0.0044651, -0.0074559, -0.0042787, -0.0021251, 0.0017806
9: -0.0036245, -0.0033504, -0.0036406, -0.0033665, -0.0001536, 0.0001833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004715, upper bound: 0.0004672
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004714, upper bound: 0.0004732
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0089727, -0.0062092, -0.0089727, -0.0062092, -0.0015168, 0.0015168
1: -0.0054684, -0.0046893, -0.0054684, -0.0046893, -0.0004276, 0.0004276
2: -0.0017872, 0.0039615, -0.0017872, 0.0039615, -0.0031552, 0.0031552
3: 0.0013908, 0.0021515, 0.0013908, 0.0021515, -0.0004175, 0.0004175
4: 0.0031313, 0.0074275, 0.0031313, 0.0074275, -0.0023580, 0.0023580
5: 0.9963762, 0.9975698, 0.9963762, 0.9975698, -0.0006551, 0.0006551
6: 0.0045943, 0.0056778, 0.0045943, 0.0056778, -0.0005947, 0.0005947
7: -0.0062362, -0.0021930, -0.0062362, -0.0021930, -0.0022191, 0.0022191
8: -0.0074860, -0.0043392, -0.0074860, -0.0043392, -0.0017272, 0.0017272
9: -0.0036354, -0.0033639, -0.0036354, -0.0033639, -0.0001490, 0.0001490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004802, upper bound: 0.0004721
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004802, upper bound: 0.0004794
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089727, -0.0062092, -0.0089462, -0.0061560, -0.0016523, 0.0015891
1: -0.0054684, -0.0046893, -0.0054609, -0.0046743, -0.0004658, 0.0004480
2: -0.0017872, 0.0039615, -0.0017322, 0.0040721, -0.0034371, 0.0033056
3: 0.0013908, 0.0021515, 0.0013981, 0.0021662, -0.0004549, 0.0004374
4: 0.0031313, 0.0074275, 0.0030487, 0.0073864, -0.0024704, 0.0025687
5: 0.9963762, 0.9975698, 0.9963533, 0.9975584, -0.0006864, 0.0007137
6: 0.0045943, 0.0056778, 0.0045735, 0.0056674, -0.0006230, 0.0006478
7: -0.0062362, -0.0021930, -0.0063140, -0.0022317, -0.0023249, 0.0024174
8: -0.0074860, -0.0043392, -0.0074559, -0.0042787, -0.0018815, 0.0018095
9: -0.0036354, -0.0033639, -0.0036406, -0.0033665, -0.0001561, 0.0001623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004802, upper bound: 0.0004732
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004802, upper bound: 0.0004819
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0089322, -0.0061583, -0.0090544, -0.0063208, -0.0015483, 0.0017559
1: -0.0054570, -0.0046749, -0.0054914, -0.0047207, -0.0004365, 0.0004950
2: -0.0017030, 0.0040673, -0.0019572, 0.0037294, -0.0032208, 0.0036526
3: 0.0014019, 0.0021655, 0.0013683, 0.0021208, -0.0004262, 0.0004834
4: 0.0030522, 0.0073646, 0.0033048, 0.0075546, -0.0027297, 0.0024070
5: 0.9963543, 0.9975524, 0.9964244, 0.9976051, -0.0007584, 0.0006687
6: 0.0045744, 0.0056619, 0.0046381, 0.0057098, -0.0006884, 0.0006070
7: -0.0063107, -0.0022522, -0.0060730, -0.0020734, -0.0025690, 0.0022653
8: -0.0074400, -0.0042813, -0.0075791, -0.0044662, -0.0017631, 0.0019994
9: -0.0036404, -0.0033679, -0.0036244, -0.0033558, -0.0001725, 0.0001521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004099, upper bound: 0.0004064
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004083, upper bound: 0.0003968
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0089318, -0.0061585, -0.0090735, -0.0062855, -0.0015796, 0.0017675
1: -0.0054569, -0.0046750, -0.0054968, -0.0047108, -0.0004453, 0.0004983
2: -0.0017020, 0.0040670, -0.0019969, 0.0038027, -0.0032858, 0.0036767
3: 0.0014021, 0.0021655, 0.0013630, 0.0021305, -0.0004348, 0.0004865
4: 0.0030525, 0.0073639, 0.0032500, 0.0075842, -0.0027477, 0.0024556
5: 0.9963543, 0.9975522, 0.9964092, 0.9976133, -0.0007634, 0.0006822
6: 0.0045745, 0.0056617, 0.0046243, 0.0057173, -0.0006929, 0.0006193
7: -0.0063104, -0.0022529, -0.0061245, -0.0020455, -0.0025859, 0.0023110
8: -0.0074394, -0.0042815, -0.0076008, -0.0044261, -0.0017987, 0.0020126
9: -0.0036404, -0.0033679, -0.0036279, -0.0033540, -0.0001736, 0.0001552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004126, upper bound: 0.0004056
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004107, upper bound: 0.0003951
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0089322, -0.0061583, -0.0090264, -0.0062699, -0.0014912, 0.0016263
1: -0.0054570, -0.0046749, -0.0054835, -0.0047064, -0.0004204, 0.0004585
2: -0.0017030, 0.0040673, -0.0018990, 0.0038352, -0.0031019, 0.0033830
3: 0.0014019, 0.0021655, 0.0013760, 0.0021348, -0.0004105, 0.0004477
4: 0.0030522, 0.0073646, 0.0032257, 0.0075111, -0.0025283, 0.0023182
5: 0.9963543, 0.9975524, 0.9964024, 0.9975930, -0.0007024, 0.0006441
6: 0.0045744, 0.0056619, 0.0046181, 0.0056988, -0.0006376, 0.0005846
7: -0.0063107, -0.0022522, -0.0061474, -0.0021144, -0.0023794, 0.0021817
8: -0.0074400, -0.0042813, -0.0075472, -0.0044083, -0.0016980, 0.0018519
9: -0.0036404, -0.0033679, -0.0036294, -0.0033586, -0.0001598, 0.0001465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004126, upper bound: 0.0004122
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004111, upper bound: 0.0004035
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0089318, -0.0061585, -0.0090446, -0.0062360, -0.0015214, 0.0016386
1: -0.0054569, -0.0046750, -0.0054887, -0.0046968, -0.0004289, 0.0004620
2: -0.0017020, 0.0040670, -0.0019368, 0.0039057, -0.0031648, 0.0034085
3: 0.0014021, 0.0021655, 0.0013710, 0.0021442, -0.0004188, 0.0004511
4: 0.0030525, 0.0073639, 0.0031730, 0.0075393, -0.0025473, 0.0023652
5: 0.9963543, 0.9975522, 0.9963878, 0.9976009, -0.0007077, 0.0006571
6: 0.0045745, 0.0056617, 0.0046048, 0.0057060, -0.0006424, 0.0005965
7: -0.0063104, -0.0022529, -0.0061970, -0.0020878, -0.0023973, 0.0022259
8: -0.0074394, -0.0042815, -0.0075679, -0.0043697, -0.0017324, 0.0018658
9: -0.0036404, -0.0033679, -0.0036327, -0.0033568, -0.0001610, 0.0001495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004149, upper bound: 0.0004118
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004135, upper bound: 0.0004024
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090809, -0.0062690, -0.0089727, -0.0062092, -0.0018090, 0.0016361
1: -0.0054989, -0.0047061, -0.0054684, -0.0046893, -0.0005100, 0.0004613
2: -0.0020122, 0.0038371, -0.0017872, 0.0039615, -0.0037632, 0.0034035
3: 0.0013610, 0.0021351, 0.0013908, 0.0021515, -0.0004980, 0.0004504
4: 0.0032242, 0.0075956, 0.0031313, 0.0074275, -0.0025435, 0.0028124
5: 0.9964020, 0.9976165, 0.9963762, 0.9975698, -0.0007067, 0.0007814
6: 0.0046178, 0.0057202, 0.0045943, 0.0056778, -0.0006414, 0.0007092
7: -0.0061488, -0.0020348, -0.0062362, -0.0021930, -0.0023938, 0.0026467
8: -0.0076092, -0.0044073, -0.0074860, -0.0043392, -0.0020600, 0.0018631
9: -0.0036295, -0.0033533, -0.0036354, -0.0033639, -0.0001607, 0.0001777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004717, upper bound: 0.0004651
time: 1.00 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004718, upper bound: 0.0004705
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0089462, -0.0061560, -0.0089727, -0.0062092, -0.0015891, 0.0016523
1: -0.0054609, -0.0046743, -0.0054684, -0.0046893, -0.0004480, 0.0004658
2: -0.0017322, 0.0040721, -0.0017872, 0.0039615, -0.0033056, 0.0034371
3: 0.0013981, 0.0021662, 0.0013908, 0.0021515, -0.0004374, 0.0004549
4: 0.0030487, 0.0073864, 0.0031313, 0.0074275, -0.0025687, 0.0024704
5: 0.9963533, 0.9975584, 0.9963762, 0.9975698, -0.0007137, 0.0006864
6: 0.0045735, 0.0056674, 0.0045943, 0.0056778, -0.0006478, 0.0006230
7: -0.0063140, -0.0022317, -0.0062362, -0.0021930, -0.0024174, 0.0023249
8: -0.0074559, -0.0042787, -0.0074860, -0.0043392, -0.0018095, 0.0018815
9: -0.0036406, -0.0033665, -0.0036354, -0.0033639, -0.0001623, 0.0001561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004635, upper bound: 0.0004801
time: 1.16 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004717, upper bound: 0.0004800
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090809, -0.0062690, -0.0089462, -0.0061560, -0.0017417, 0.0015060
1: -0.0054989, -0.0047061, -0.0054609, -0.0046743, -0.0004910, 0.0004246
2: -0.0020122, 0.0038371, -0.0017322, 0.0040721, -0.0036230, 0.0031329
3: 0.0013610, 0.0021351, 0.0013981, 0.0021662, -0.0004794, 0.0004146
4: 0.0032242, 0.0075956, 0.0030487, 0.0073864, -0.0023413, 0.0027076
5: 0.9964020, 0.9976165, 0.9963533, 0.9975584, -0.0006505, 0.0007523
6: 0.0046178, 0.0057202, 0.0045735, 0.0056674, -0.0005904, 0.0006828
7: -0.0061488, -0.0020348, -0.0063140, -0.0022317, -0.0022034, 0.0025482
8: -0.0076092, -0.0044073, -0.0074559, -0.0042787, -0.0019832, 0.0017149
9: -0.0036295, -0.0033533, -0.0036406, -0.0033665, -0.0001480, 0.0001711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004719, upper bound: 0.0004651
time: 1.13 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004719, upper bound: 0.0004704
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089462, -0.0061560, -0.0089462, -0.0061560, -0.0015242, 0.0015242
1: -0.0054609, -0.0046743, -0.0054609, -0.0046743, -0.0004297, 0.0004297
2: -0.0017322, 0.0040721, -0.0017322, 0.0040721, -0.0031707, 0.0031707
3: 0.0013981, 0.0021662, 0.0013981, 0.0021662, -0.0004196, 0.0004196
4: 0.0030487, 0.0073864, 0.0030487, 0.0073864, -0.0023696, 0.0023696
5: 0.9963533, 0.9975584, 0.9963533, 0.9975584, -0.0006583, 0.0006583
6: 0.0045735, 0.0056674, 0.0045735, 0.0056674, -0.0005976, 0.0005976
7: -0.0063140, -0.0022317, -0.0063140, -0.0022317, -0.0022300, 0.0022300
8: -0.0074559, -0.0042787, -0.0074559, -0.0042787, -0.0017356, 0.0017356
9: -0.0036406, -0.0033665, -0.0036406, -0.0033665, -0.0001497, 0.0001497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004719, upper bound: 0.0004724
time: 1.08 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004719, upper bound: 0.0004798
time: 0.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004048, upper bound: 0.0004064
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004019, upper bound: 0.0003966
IS_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004068, upper bound: 0.0004055
IS_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004030, upper bound: 0.0003949
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004048, upper bound: 0.0004102
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004017, upper bound: 0.0003986
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004067, upper bound: 0.0004097
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004030, upper bound: 0.0003975
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004715, upper bound: 0.0004648
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004714, upper bound: 0.0004701
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004715, upper bound: 0.0004672
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004714, upper bound: 0.0004732
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004802, upper bound: 0.0004721
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004802, upper bound: 0.0004794
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004802, upper bound: 0.0004732
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004802, upper bound: 0.0004819
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004099, upper bound: 0.0004064
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004083, upper bound: 0.0003968
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004126, upper bound: 0.0004056
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004107, upper bound: 0.0003951
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004126, upper bound: 0.0004122
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004111, upper bound: 0.0004035
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004149, upper bound: 0.0004118
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004135, upper bound: 0.0004024
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004717, upper bound: 0.0004651
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004718, upper bound: 0.0004705
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004635, upper bound: 0.0004801
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004717, upper bound: 0.0004800
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004719, upper bound: 0.0004651
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004719, upper bound: 0.0004704
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004719, upper bound: 0.0004724
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 5, lower bound: -0.0004719, upper bound: 0.0004798

## BFS IS instance: IS_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089574, -0.0062489, -0.0090544, -0.0063208, -0.0014776, 0.0015826
1: -0.0054641, -0.0047005, -0.0054914, -0.0047207, -0.0004166, 0.0004462
2: -0.0017553, 0.0038789, -0.0019572, 0.0037294, -0.0030736, 0.0032922
3: 0.0013950, 0.0021406, 0.0013683, 0.0021208, -0.0004067, 0.0004357
4: 0.0031930, 0.0074036, 0.0033048, 0.0075546, -0.0024604, 0.0022970
5: 0.9963934, 0.9975632, 0.9964244, 0.9976051, -0.0006836, 0.0006382
6: 0.0046099, 0.0056718, 0.0046381, 0.0057098, -0.0006205, 0.0005793
7: -0.0061781, -0.0022155, -0.0060730, -0.0020734, -0.0023155, 0.0021618
8: -0.0074685, -0.0043844, -0.0075791, -0.0044662, -0.0016825, 0.0018022
9: -0.0036315, -0.0033654, -0.0036244, -0.0033558, -0.0001555, 0.0001452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003923, upper bound: 0.0003930
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003923, upper bound: 0.0003934
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090894, -0.0063038, -0.0090537, -0.0063525, -0.0016260, 0.0016010
1: -0.0055013, -0.0047159, -0.0054912, -0.0047297, -0.0004584, 0.0004514
2: -0.0020299, 0.0037646, -0.0019556, 0.0036634, -0.0033823, 0.0033304
3: 0.0013587, 0.0021255, 0.0013685, 0.0021121, -0.0004476, 0.0004407
4: 0.0032784, 0.0076089, 0.0033540, 0.0075534, -0.0024889, 0.0025277
5: 0.9964171, 0.9976202, 0.9964381, 0.9976048, -0.0006915, 0.0007023
6: 0.0046314, 0.0057235, 0.0046505, 0.0057095, -0.0006277, 0.0006375
7: -0.0060978, -0.0020223, -0.0060266, -0.0020746, -0.0023423, 0.0023789
8: -0.0076189, -0.0044470, -0.0075782, -0.0045023, -0.0018515, 0.0018231
9: -0.0036261, -0.0033524, -0.0036213, -0.0033559, -0.0001573, 0.0001597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003825
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003841
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089567, -0.0062491, -0.0090735, -0.0062855, -0.0015109, 0.0015943
1: -0.0054639, -0.0047005, -0.0054968, -0.0047108, -0.0004260, 0.0004495
2: -0.0017539, 0.0038785, -0.0019969, 0.0038027, -0.0031430, 0.0033164
3: 0.0013952, 0.0021406, 0.0013630, 0.0021305, -0.0004159, 0.0004389
4: 0.0031933, 0.0074026, 0.0032500, 0.0075842, -0.0024785, 0.0023489
5: 0.9963934, 0.9975629, 0.9964092, 0.9976133, -0.0006886, 0.0006526
6: 0.0046100, 0.0056715, 0.0046243, 0.0057173, -0.0006250, 0.0005924
7: -0.0061779, -0.0022164, -0.0061245, -0.0020455, -0.0023325, 0.0022106
8: -0.0074678, -0.0043846, -0.0076008, -0.0044261, -0.0017205, 0.0018154
9: -0.0036315, -0.0033654, -0.0036279, -0.0033540, -0.0001566, 0.0001484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003940, upper bound: 0.0003920
time: 0.98 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003940, upper bound: 0.0003920
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090890, -0.0063040, -0.0090727, -0.0063202, -0.0016530, 0.0016106
1: -0.0055012, -0.0047160, -0.0054966, -0.0047206, -0.0004660, 0.0004541
2: -0.0020291, 0.0037643, -0.0019952, 0.0037306, -0.0034385, 0.0033504
3: 0.0013588, 0.0021254, 0.0013633, 0.0021210, -0.0004550, 0.0004434
4: 0.0032787, 0.0076083, 0.0033039, 0.0075829, -0.0025039, 0.0025697
5: 0.9964172, 0.9976200, 0.9964241, 0.9976130, -0.0006956, 0.0007140
6: 0.0046315, 0.0057234, 0.0046379, 0.0057170, -0.0006314, 0.0006481
7: -0.0060975, -0.0020229, -0.0060738, -0.0020467, -0.0023564, 0.0024184
8: -0.0076185, -0.0044471, -0.0075999, -0.0044656, -0.0018823, 0.0018340
9: -0.0036261, -0.0033525, -0.0036245, -0.0033541, -0.0001582, 0.0001624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003902, upper bound: 0.0003802
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003903, upper bound: 0.0003816
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089574, -0.0062489, -0.0090264, -0.0062699, -0.0016164, 0.0016512
1: -0.0054641, -0.0047005, -0.0054835, -0.0047064, -0.0004557, 0.0004655
2: -0.0017553, 0.0038789, -0.0018990, 0.0038352, -0.0033624, 0.0034349
3: 0.0013950, 0.0021406, 0.0013760, 0.0021348, -0.0004450, 0.0004545
4: 0.0031930, 0.0074036, 0.0032257, 0.0075111, -0.0025670, 0.0025129
5: 0.9963934, 0.9975632, 0.9964024, 0.9975930, -0.0007132, 0.0006982
6: 0.0046099, 0.0056718, 0.0046181, 0.0056988, -0.0006474, 0.0006337
7: -0.0061781, -0.0022155, -0.0061474, -0.0021144, -0.0024158, 0.0023649
8: -0.0074685, -0.0043844, -0.0075472, -0.0044083, -0.0018406, 0.0018802
9: -0.0036315, -0.0033654, -0.0036294, -0.0033586, -0.0001622, 0.0001588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003923, upper bound: 0.0003968
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003924, upper bound: 0.0003972
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090894, -0.0063038, -0.0090257, -0.0063017, -0.0017572, 0.0016697
1: -0.0055013, -0.0047159, -0.0054833, -0.0047153, -0.0004954, 0.0004707
2: -0.0020299, 0.0037646, -0.0018974, 0.0037690, -0.0036554, 0.0034733
3: 0.0013587, 0.0021255, 0.0013762, 0.0021261, -0.0004837, 0.0004596
4: 0.0032784, 0.0076089, 0.0032751, 0.0075099, -0.0025957, 0.0027318
5: 0.9964171, 0.9976202, 0.9964162, 0.9975927, -0.0007212, 0.0007590
6: 0.0046314, 0.0057235, 0.0046306, 0.0056986, -0.0006546, 0.0006889
7: -0.0060978, -0.0020223, -0.0061009, -0.0021155, -0.0024429, 0.0025709
8: -0.0076189, -0.0044470, -0.0075464, -0.0044446, -0.0020010, 0.0019013
9: -0.0036261, -0.0033524, -0.0036263, -0.0033587, -0.0001640, 0.0001726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003842
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003862
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089567, -0.0062491, -0.0090446, -0.0062360, -0.0016576, 0.0016636
1: -0.0054639, -0.0047005, -0.0054887, -0.0046968, -0.0004673, 0.0004690
2: -0.0017539, 0.0038785, -0.0019368, 0.0039057, -0.0034481, 0.0034607
3: 0.0013952, 0.0021406, 0.0013710, 0.0021442, -0.0004563, 0.0004580
4: 0.0031933, 0.0074026, 0.0031730, 0.0075393, -0.0025863, 0.0025769
5: 0.9963934, 0.9975629, 0.9963878, 0.9976009, -0.0007185, 0.0007159
6: 0.0046100, 0.0056715, 0.0046048, 0.0057060, -0.0006522, 0.0006499
7: -0.0061779, -0.0022164, -0.0061970, -0.0020878, -0.0024340, 0.0024252
8: -0.0074678, -0.0043846, -0.0075679, -0.0043697, -0.0018875, 0.0018944
9: -0.0036315, -0.0033654, -0.0036327, -0.0033568, -0.0001634, 0.0001628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003940, upper bound: 0.0003963
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003941, upper bound: 0.0003965
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090890, -0.0063040, -0.0090438, -0.0062711, -0.0017948, 0.0016800
1: -0.0055012, -0.0047160, -0.0054884, -0.0047067, -0.0005060, 0.0004737
2: -0.0020291, 0.0037643, -0.0019351, 0.0038327, -0.0037336, 0.0034948
3: 0.0013588, 0.0021254, 0.0013712, 0.0021345, -0.0004941, 0.0004625
4: 0.0032787, 0.0076083, 0.0032276, 0.0075381, -0.0026118, 0.0027902
5: 0.9964172, 0.9976200, 0.9964029, 0.9976006, -0.0007256, 0.0007752
6: 0.0046315, 0.0057234, 0.0046186, 0.0057057, -0.0006587, 0.0007037
7: -0.0060975, -0.0020229, -0.0061456, -0.0020890, -0.0024580, 0.0026259
8: -0.0076185, -0.0044471, -0.0075670, -0.0044097, -0.0020438, 0.0019131
9: -0.0036261, -0.0033525, -0.0036293, -0.0033569, -0.0001650, 0.0001763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003903, upper bound: 0.0003829
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003902, upper bound: 0.0003848
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090544, -0.0063208, -0.0089589, -0.0062095, -0.0016858, 0.0014858
1: -0.0054914, -0.0047207, -0.0054645, -0.0046893, -0.0004753, 0.0004189
2: -0.0019572, 0.0037294, -0.0017584, 0.0039609, -0.0035067, 0.0030907
3: 0.0013683, 0.0021208, 0.0013946, 0.0021515, -0.0004641, 0.0004090
4: 0.0033048, 0.0075546, 0.0031317, 0.0074060, -0.0023098, 0.0026207
5: 0.9964244, 0.9976051, 0.9963763, 0.9975638, -0.0006417, 0.0007281
6: 0.0046381, 0.0057098, 0.0045944, 0.0056724, -0.0005825, 0.0006609
7: -0.0060730, -0.0020734, -0.0062358, -0.0022133, -0.0021738, 0.0024664
8: -0.0075791, -0.0044662, -0.0074703, -0.0043395, -0.0019196, 0.0016918
9: -0.0036244, -0.0033558, -0.0036353, -0.0033652, -0.0001460, 0.0001656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004064, upper bound: 0.0004048
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003967, upper bound: 0.0004019
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090735, -0.0062855, -0.0089582, -0.0062096, -0.0016974, 0.0015162
1: -0.0054968, -0.0047108, -0.0054643, -0.0046894, -0.0004786, 0.0004275
2: -0.0019969, 0.0038027, -0.0017571, 0.0039606, -0.0035309, 0.0031539
3: 0.0013630, 0.0021305, 0.0013948, 0.0021514, -0.0004673, 0.0004174
4: 0.0032500, 0.0075842, 0.0031320, 0.0074050, -0.0023570, 0.0026388
5: 0.9964092, 0.9976133, 0.9963763, 0.9975636, -0.0006549, 0.0007331
6: 0.0046243, 0.0057173, 0.0045945, 0.0056721, -0.0005944, 0.0006655
7: -0.0061245, -0.0020455, -0.0062356, -0.0022142, -0.0022182, 0.0024834
8: -0.0076008, -0.0044261, -0.0074695, -0.0043397, -0.0019328, 0.0017265
9: -0.0036279, -0.0033540, -0.0036353, -0.0033653, -0.0001490, 0.0001668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004055, upper bound: 0.0004068
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003948, upper bound: 0.0004030
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090544, -0.0063208, -0.0089325, -0.0061563, -0.0018152, 0.0015483
1: -0.0054914, -0.0047207, -0.0054571, -0.0046743, -0.0005118, 0.0004365
2: -0.0019572, 0.0037294, -0.0017036, 0.0040715, -0.0037760, 0.0032208
3: 0.0013683, 0.0021208, 0.0014019, 0.0021661, -0.0004997, 0.0004262
4: 0.0033048, 0.0075546, 0.0030491, 0.0073650, -0.0024071, 0.0028220
5: 0.9964244, 0.9976051, 0.9963534, 0.9975525, -0.0006688, 0.0007840
6: 0.0046381, 0.0057098, 0.0045736, 0.0056620, -0.0006070, 0.0007117
7: -0.0060730, -0.0020734, -0.0063136, -0.0022518, -0.0022653, 0.0026558
8: -0.0075791, -0.0044662, -0.0074403, -0.0042790, -0.0020670, 0.0017631
9: -0.0036244, -0.0033558, -0.0036406, -0.0033678, -0.0001521, 0.0001783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004063, upper bound: 0.0004099
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003968, upper bound: 0.0004083
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090735, -0.0062855, -0.0089320, -0.0061565, -0.0018268, 0.0015796
1: -0.0054968, -0.0047108, -0.0054569, -0.0046744, -0.0005150, 0.0004453
2: -0.0019969, 0.0038027, -0.0017026, 0.0040711, -0.0038001, 0.0032859
3: 0.0013630, 0.0021305, 0.0014020, 0.0021660, -0.0005029, 0.0004348
4: 0.0032500, 0.0075842, 0.0030493, 0.0073643, -0.0024556, 0.0028400
5: 0.9964092, 0.9976133, 0.9963534, 0.9975522, -0.0006823, 0.0007890
6: 0.0046243, 0.0057173, 0.0045737, 0.0056618, -0.0006193, 0.0007162
7: -0.0061245, -0.0020455, -0.0063134, -0.0022525, -0.0023110, 0.0026727
8: -0.0076008, -0.0044261, -0.0074397, -0.0042792, -0.0020802, 0.0017987
9: -0.0036279, -0.0033540, -0.0036405, -0.0033679, -0.0001552, 0.0001795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004055, upper bound: 0.0004127
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003950, upper bound: 0.0004107
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089224, -0.0062102, -0.0089589, -0.0062095, -0.0014669, 0.0015019
1: -0.0054542, -0.0046895, -0.0054645, -0.0046893, -0.0004136, 0.0004234
2: -0.0016825, 0.0039594, -0.0017584, 0.0039609, -0.0030514, 0.0031242
3: 0.0014046, 0.0021513, 0.0013946, 0.0021515, -0.0004038, 0.0004134
4: 0.0031328, 0.0073493, 0.0031317, 0.0074060, -0.0023348, 0.0022804
5: 0.9963766, 0.9975481, 0.9963763, 0.9975638, -0.0006487, 0.0006336
6: 0.0045947, 0.0056581, 0.0045944, 0.0056724, -0.0005888, 0.0005751
7: -0.0062348, -0.0022666, -0.0062358, -0.0022133, -0.0021973, 0.0021461
8: -0.0074287, -0.0043403, -0.0074703, -0.0043395, -0.0016703, 0.0017102
9: -0.0036353, -0.0033688, -0.0036353, -0.0033652, -0.0001475, 0.0001441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004530
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004529
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0089309, -0.0061866, -0.0089582, -0.0062096, -0.0014756, 0.0015304
1: -0.0054566, -0.0046829, -0.0054643, -0.0046894, -0.0004160, 0.0004315
2: -0.0017002, 0.0040085, -0.0017571, 0.0039606, -0.0030697, 0.0031836
3: 0.0014023, 0.0021578, 0.0013948, 0.0021514, -0.0004062, 0.0004213
4: 0.0030962, 0.0073625, 0.0031320, 0.0074050, -0.0023792, 0.0022941
5: 0.9963664, 0.9975518, 0.9963763, 0.9975636, -0.0006610, 0.0006374
6: 0.0045855, 0.0056614, 0.0045945, 0.0056721, -0.0006000, 0.0005785
7: -0.0062693, -0.0022542, -0.0062356, -0.0022142, -0.0022391, 0.0021590
8: -0.0074384, -0.0043135, -0.0074695, -0.0043397, -0.0016803, 0.0017427
9: -0.0036376, -0.0033680, -0.0036353, -0.0033653, -0.0001504, 0.0001450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004600
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004600
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089224, -0.0062102, -0.0089325, -0.0061563, -0.0016024, 0.0015726
1: -0.0054542, -0.0046895, -0.0054571, -0.0046743, -0.0004518, 0.0004434
2: -0.0016825, 0.0039594, -0.0017036, 0.0040715, -0.0033333, 0.0032714
3: 0.0014046, 0.0021513, 0.0014019, 0.0021661, -0.0004411, 0.0004329
4: 0.0031328, 0.0073493, 0.0030491, 0.0073650, -0.0024448, 0.0024911
5: 0.9963766, 0.9975481, 0.9963534, 0.9975525, -0.0006792, 0.0006921
6: 0.0045947, 0.0056581, 0.0045736, 0.0056620, -0.0006165, 0.0006282
7: -0.0062348, -0.0022666, -0.0063136, -0.0022518, -0.0023008, 0.0023444
8: -0.0074287, -0.0043403, -0.0074403, -0.0042790, -0.0018246, 0.0017908
9: -0.0036353, -0.0033688, -0.0036406, -0.0033678, -0.0001545, 0.0001574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004533
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004533
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089309, -0.0061866, -0.0089320, -0.0061565, -0.0016112, 0.0016047
1: -0.0054566, -0.0046829, -0.0054569, -0.0046744, -0.0004542, 0.0004524
2: -0.0017002, 0.0040085, -0.0017026, 0.0040711, -0.0033515, 0.0033381
3: 0.0014023, 0.0021578, 0.0014020, 0.0021660, -0.0004435, 0.0004417
4: 0.0030962, 0.0073625, 0.0030493, 0.0073643, -0.0024947, 0.0025047
5: 0.9963664, 0.9975518, 0.9963534, 0.9975522, -0.0006931, 0.0006959
6: 0.0045855, 0.0056614, 0.0045737, 0.0056618, -0.0006291, 0.0006317
7: -0.0062693, -0.0022542, -0.0063134, -0.0022525, -0.0023478, 0.0023572
8: -0.0074384, -0.0043135, -0.0074397, -0.0042792, -0.0018346, 0.0018273
9: -0.0036376, -0.0033680, -0.0036405, -0.0033679, -0.0001576, 0.0001583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004616
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004616
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089309, -0.0061956, -0.0090544, -0.0063208, -0.0015463, 0.0017176
1: -0.0054566, -0.0046854, -0.0054914, -0.0047207, -0.0004360, 0.0004843
2: -0.0017003, 0.0039898, -0.0019572, 0.0037294, -0.0032167, 0.0035730
3: 0.0014023, 0.0021553, 0.0013683, 0.0021208, -0.0004257, 0.0004728
4: 0.0031102, 0.0073626, 0.0033048, 0.0075546, -0.0026703, 0.0024039
5: 0.9963703, 0.9975517, 0.9964244, 0.9976051, -0.0007419, 0.0006679
6: 0.0045890, 0.0056614, 0.0046381, 0.0057098, -0.0006734, 0.0006062
7: -0.0062561, -0.0022541, -0.0060730, -0.0020734, -0.0025130, 0.0022624
8: -0.0074385, -0.0043237, -0.0075791, -0.0044662, -0.0017608, 0.0019559
9: -0.0036367, -0.0033680, -0.0036244, -0.0033558, -0.0001687, 0.0001519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003971, upper bound: 0.0003930
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003972, upper bound: 0.0003934
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090657, -0.0062490, -0.0090537, -0.0063525, -0.0016781, 0.0017362
1: -0.0054946, -0.0047005, -0.0054912, -0.0047297, -0.0004731, 0.0004895
2: -0.0019807, 0.0038787, -0.0019556, 0.0036634, -0.0034908, 0.0036116
3: 0.0013652, 0.0021406, 0.0013685, 0.0021121, -0.0004620, 0.0004779
4: 0.0031932, 0.0075721, 0.0033540, 0.0075534, -0.0026991, 0.0026088
5: 0.9963934, 0.9976100, 0.9964381, 0.9976048, -0.0007499, 0.0007248
6: 0.0046099, 0.0057142, 0.0046505, 0.0057095, -0.0006807, 0.0006579
7: -0.0061780, -0.0020569, -0.0060266, -0.0020746, -0.0025402, 0.0024552
8: -0.0075919, -0.0043845, -0.0075782, -0.0045023, -0.0019109, 0.0019770
9: -0.0036315, -0.0033547, -0.0036213, -0.0033559, -0.0001706, 0.0001649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003958, upper bound: 0.0003826
time: 1.01 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003958, upper bound: 0.0003844
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089304, -0.0061958, -0.0090735, -0.0062855, -0.0015776, 0.0017292
1: -0.0054565, -0.0046855, -0.0054968, -0.0047108, -0.0004448, 0.0004875
2: -0.0016993, 0.0039894, -0.0019969, 0.0038027, -0.0032817, 0.0035971
3: 0.0014024, 0.0021552, 0.0013630, 0.0021305, -0.0004343, 0.0004760
4: 0.0031104, 0.0073618, 0.0032500, 0.0075842, -0.0026883, 0.0024525
5: 0.9963704, 0.9975516, 0.9964092, 0.9976133, -0.0007469, 0.0006814
6: 0.0045891, 0.0056612, 0.0046243, 0.0057173, -0.0006779, 0.0006185
7: -0.0062559, -0.0022549, -0.0061245, -0.0020455, -0.0025300, 0.0023081
8: -0.0074379, -0.0043239, -0.0076008, -0.0044261, -0.0017964, 0.0019691
9: -0.0036367, -0.0033680, -0.0036279, -0.0033540, -0.0001699, 0.0001550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004000, upper bound: 0.0003919
time: 1.03 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004000, upper bound: 0.0003920
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090639, -0.0062492, -0.0090727, -0.0063202, -0.0017057, 0.0017458
1: -0.0054941, -0.0047005, -0.0054966, -0.0047206, -0.0004809, 0.0004922
2: -0.0019768, 0.0038783, -0.0019952, 0.0037306, -0.0035483, 0.0036315
3: 0.0013657, 0.0021405, 0.0013633, 0.0021210, -0.0004696, 0.0004806
4: 0.0031935, 0.0075692, 0.0033039, 0.0075829, -0.0027140, 0.0026518
5: 0.9963934, 0.9976093, 0.9964241, 0.9976130, -0.0007540, 0.0007367
6: 0.0046100, 0.0057135, 0.0046379, 0.0057170, -0.0006844, 0.0006687
7: -0.0061778, -0.0020596, -0.0060738, -0.0020467, -0.0025542, 0.0024956
8: -0.0075898, -0.0043847, -0.0075999, -0.0044656, -0.0019423, 0.0019879
9: -0.0036314, -0.0033549, -0.0036245, -0.0033541, -0.0001715, 0.0001676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003983, upper bound: 0.0003806
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003983, upper bound: 0.0003821
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089309, -0.0061956, -0.0090264, -0.0062699, -0.0014891, 0.0015868
1: -0.0054566, -0.0046854, -0.0054835, -0.0047064, -0.0004198, 0.0004474
2: -0.0017003, 0.0039898, -0.0018990, 0.0038352, -0.0030976, 0.0033009
3: 0.0014023, 0.0021553, 0.0013760, 0.0021348, -0.0004099, 0.0004368
4: 0.0031102, 0.0073626, 0.0032257, 0.0075111, -0.0024669, 0.0023150
5: 0.9963703, 0.9975517, 0.9964024, 0.9975930, -0.0006854, 0.0006432
6: 0.0045890, 0.0056614, 0.0046181, 0.0056988, -0.0006221, 0.0005838
7: -0.0062561, -0.0022541, -0.0061474, -0.0021144, -0.0023216, 0.0021787
8: -0.0074385, -0.0043237, -0.0075472, -0.0044083, -0.0016957, 0.0018069
9: -0.0036367, -0.0033680, -0.0036294, -0.0033586, -0.0001559, 0.0001463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004001, upper bound: 0.0003986
time: 0.94 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004001, upper bound: 0.0003993
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090657, -0.0062490, -0.0090257, -0.0063017, -0.0016360, 0.0016074
1: -0.0054946, -0.0047005, -0.0054833, -0.0047153, -0.0004612, 0.0004532
2: -0.0019807, 0.0038787, -0.0018974, 0.0037690, -0.0034032, 0.0033436
3: 0.0013652, 0.0021406, 0.0013762, 0.0021261, -0.0004504, 0.0004425
4: 0.0031932, 0.0075721, 0.0032751, 0.0075099, -0.0024988, 0.0025433
5: 0.9963934, 0.9976100, 0.9964162, 0.9975927, -0.0006942, 0.0007066
6: 0.0046099, 0.0057142, 0.0046306, 0.0056986, -0.0006302, 0.0006414
7: -0.0061780, -0.0020569, -0.0061009, -0.0021155, -0.0023517, 0.0023935
8: -0.0075919, -0.0043845, -0.0075464, -0.0044446, -0.0018629, 0.0018303
9: -0.0036315, -0.0033547, -0.0036263, -0.0033587, -0.0001579, 0.0001607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003986, upper bound: 0.0003892
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003986, upper bound: 0.0003910
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089304, -0.0061958, -0.0090446, -0.0062360, -0.0015192, 0.0015991
1: -0.0054565, -0.0046855, -0.0054887, -0.0046968, -0.0004283, 0.0004508
2: -0.0016993, 0.0039894, -0.0019368, 0.0039057, -0.0031603, 0.0033265
3: 0.0014024, 0.0021552, 0.0013710, 0.0021442, -0.0004182, 0.0004402
4: 0.0031104, 0.0073618, 0.0031730, 0.0075393, -0.0024860, 0.0023618
5: 0.9963704, 0.9975516, 0.9963878, 0.9976009, -0.0006907, 0.0006562
6: 0.0045891, 0.0056612, 0.0046048, 0.0057060, -0.0006269, 0.0005956
7: -0.0062559, -0.0022549, -0.0061970, -0.0020878, -0.0023396, 0.0022227
8: -0.0074379, -0.0043239, -0.0075679, -0.0043697, -0.0017299, 0.0018209
9: -0.0036367, -0.0033680, -0.0036327, -0.0033568, -0.0001571, 0.0001493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004024, upper bound: 0.0003983
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004024, upper bound: 0.0003990
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090639, -0.0062492, -0.0090438, -0.0062711, -0.0016609, 0.0016173
1: -0.0054941, -0.0047005, -0.0054884, -0.0047067, -0.0004683, 0.0004560
2: -0.0019768, 0.0038783, -0.0019351, 0.0038327, -0.0034551, 0.0033644
3: 0.0013657, 0.0021405, 0.0013712, 0.0021345, -0.0004572, 0.0004452
4: 0.0031935, 0.0075692, 0.0032276, 0.0075381, -0.0025144, 0.0025821
5: 0.9963934, 0.9976093, 0.9964029, 0.9976006, -0.0006986, 0.0007174
6: 0.0046100, 0.0057135, 0.0046186, 0.0057057, -0.0006341, 0.0006512
7: -0.0061778, -0.0020596, -0.0061456, -0.0020890, -0.0023663, 0.0024301
8: -0.0075898, -0.0043847, -0.0075670, -0.0044097, -0.0018913, 0.0018417
9: -0.0036314, -0.0033549, -0.0036293, -0.0033569, -0.0001589, 0.0001632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004009, upper bound: 0.0003878
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004009, upper bound: 0.0003901
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0090264, -0.0062699, -0.0089589, -0.0062095, -0.0017543, 0.0016212
1: -0.0054835, -0.0047064, -0.0054645, -0.0046893, -0.0004946, 0.0004571
2: -0.0018990, 0.0038352, -0.0017584, 0.0039609, -0.0036494, 0.0033724
3: 0.0013760, 0.0021348, 0.0013946, 0.0021515, -0.0004829, 0.0004463
4: 0.0032257, 0.0075111, 0.0031317, 0.0074060, -0.0025203, 0.0027273
5: 0.9964024, 0.9975930, 0.9963763, 0.9975638, -0.0007002, 0.0007577
6: 0.0046181, 0.0056988, 0.0045944, 0.0056724, -0.0006356, 0.0006878
7: -0.0061474, -0.0021144, -0.0062358, -0.0022133, -0.0023719, 0.0025667
8: -0.0075472, -0.0044083, -0.0074703, -0.0043395, -0.0019977, 0.0018460
9: -0.0036294, -0.0033586, -0.0036353, -0.0033652, -0.0001593, 0.0001724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004100, upper bound: 0.0004049
time: 0.86 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003984, upper bound: 0.0004017
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0090446, -0.0062360, -0.0089582, -0.0062096, -0.0017667, 0.0016607
1: -0.0054887, -0.0046968, -0.0054643, -0.0046894, -0.0004981, 0.0004682
2: -0.0019368, 0.0039057, -0.0017571, 0.0039606, -0.0036752, 0.0034546
3: 0.0013710, 0.0021442, 0.0013948, 0.0021514, -0.0004864, 0.0004572
4: 0.0031730, 0.0075393, 0.0031320, 0.0074050, -0.0025818, 0.0027466
5: 0.9963878, 0.9976009, 0.9963763, 0.9975636, -0.0007173, 0.0007631
6: 0.0046048, 0.0057060, 0.0045945, 0.0056721, -0.0006511, 0.0006927
7: -0.0061970, -0.0020878, -0.0062356, -0.0022142, -0.0024297, 0.0025848
8: -0.0075679, -0.0043697, -0.0074695, -0.0043397, -0.0020118, 0.0018911
9: -0.0036327, -0.0033568, -0.0036353, -0.0033653, -0.0001632, 0.0001736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004097, upper bound: 0.0004068
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003974, upper bound: 0.0004030
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0089325, -0.0061563, -0.0089224, -0.0062102, -0.0015726, 0.0016024
1: -0.0054571, -0.0046743, -0.0054542, -0.0046895, -0.0004434, 0.0004518
2: -0.0017036, 0.0040715, -0.0016825, 0.0039594, -0.0032714, 0.0033333
3: 0.0014019, 0.0021661, 0.0014046, 0.0021513, -0.0004329, 0.0004411
4: 0.0030491, 0.0073650, 0.0031328, 0.0073493, -0.0024911, 0.0024448
5: 0.9963534, 0.9975525, 0.9963766, 0.9975481, -0.0006921, 0.0006792
6: 0.0045736, 0.0056620, 0.0045947, 0.0056581, -0.0006282, 0.0006165
7: -0.0063136, -0.0022518, -0.0062348, -0.0022666, -0.0023444, 0.0023008
8: -0.0074403, -0.0042790, -0.0074287, -0.0043403, -0.0017908, 0.0018246
9: -0.0036406, -0.0033678, -0.0036353, -0.0033688, -0.0001574, 0.0001545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004563
time: 0.86 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004600
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089320, -0.0061565, -0.0089309, -0.0061866, -0.0016047, 0.0016112
1: -0.0054569, -0.0046744, -0.0054566, -0.0046829, -0.0004524, 0.0004542
2: -0.0017026, 0.0040711, -0.0017002, 0.0040085, -0.0033381, 0.0033515
3: 0.0014020, 0.0021660, 0.0014023, 0.0021578, -0.0004417, 0.0004435
4: 0.0030493, 0.0073643, 0.0030962, 0.0073625, -0.0025047, 0.0024947
5: 0.9963534, 0.9975522, 0.9963664, 0.9975518, -0.0006959, 0.0006931
6: 0.0045737, 0.0056618, 0.0045855, 0.0056614, -0.0006317, 0.0006291
7: -0.0063134, -0.0022525, -0.0062693, -0.0022542, -0.0023572, 0.0023478
8: -0.0074397, -0.0042792, -0.0074384, -0.0043135, -0.0018273, 0.0018346
9: -0.0036405, -0.0033679, -0.0036376, -0.0033680, -0.0001583, 0.0001576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004624, upper bound: 0.0004563
time: 0.89 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004625, upper bound: 0.0004600
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0090264, -0.0062699, -0.0089325, -0.0061563, -0.0016904, 0.0014912
1: -0.0054835, -0.0047064, -0.0054571, -0.0046743, -0.0004766, 0.0004204
2: -0.0018990, 0.0038352, -0.0017036, 0.0040715, -0.0035164, 0.0031019
3: 0.0013760, 0.0021348, 0.0014019, 0.0021661, -0.0004653, 0.0004105
4: 0.0032257, 0.0075111, 0.0030491, 0.0073650, -0.0023182, 0.0026279
5: 0.9964024, 0.9975930, 0.9963534, 0.9975525, -0.0006441, 0.0007301
6: 0.0046181, 0.0056988, 0.0045736, 0.0056620, -0.0005846, 0.0006627
7: -0.0061474, -0.0021144, -0.0063136, -0.0022518, -0.0021817, 0.0024732
8: -0.0075472, -0.0044083, -0.0074403, -0.0042790, -0.0019249, 0.0016980
9: -0.0036294, -0.0033586, -0.0036406, -0.0033678, -0.0001465, 0.0001661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004134, upper bound: 0.0004101
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004034, upper bound: 0.0004082
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0090446, -0.0062360, -0.0089320, -0.0061565, -0.0017027, 0.0015214
1: -0.0054887, -0.0046968, -0.0054569, -0.0046744, -0.0004800, 0.0004289
2: -0.0019368, 0.0039057, -0.0017026, 0.0040711, -0.0035419, 0.0031648
3: 0.0013710, 0.0021442, 0.0014020, 0.0021660, -0.0004687, 0.0004188
4: 0.0031730, 0.0075393, 0.0030493, 0.0073643, -0.0023651, 0.0026470
5: 0.9963878, 0.9976009, 0.9963534, 0.9975522, -0.0006571, 0.0007354
6: 0.0046048, 0.0057060, 0.0045737, 0.0056618, -0.0005965, 0.0006675
7: -0.0061970, -0.0020878, -0.0063134, -0.0022525, -0.0022259, 0.0024911
8: -0.0075679, -0.0043697, -0.0074397, -0.0042792, -0.0019388, 0.0017324
9: -0.0036327, -0.0033568, -0.0036405, -0.0033679, -0.0001495, 0.0001673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004128, upper bound: 0.0004123
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004025, upper bound: 0.0004101
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0088969, -0.0061570, -0.0089325, -0.0061563, -0.0014746, 0.0015092
1: -0.0054470, -0.0046745, -0.0054571, -0.0046743, -0.0004158, 0.0004255
2: -0.0016295, 0.0040701, -0.0017036, 0.0040715, -0.0030675, 0.0031393
3: 0.0014117, 0.0021659, 0.0014019, 0.0021661, -0.0004059, 0.0004154
4: 0.0030501, 0.0073097, 0.0030491, 0.0073650, -0.0023462, 0.0022925
5: 0.9963536, 0.9975371, 0.9963534, 0.9975525, -0.0006518, 0.0006369
6: 0.0045739, 0.0056481, 0.0045736, 0.0056620, -0.0005917, 0.0005781
7: -0.0063126, -0.0023039, -0.0063136, -0.0022518, -0.0022080, 0.0021575
8: -0.0073997, -0.0042797, -0.0074403, -0.0042790, -0.0016792, 0.0017185
9: -0.0036405, -0.0033713, -0.0036406, -0.0033678, -0.0001483, 0.0001449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004590, upper bound: 0.0004526
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004628, upper bound: 0.0004526
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0089064, -0.0061319, -0.0089320, -0.0061565, -0.0014844, 0.0015385
1: -0.0054497, -0.0046675, -0.0054569, -0.0046744, -0.0004185, 0.0004338
2: -0.0016493, 0.0041223, -0.0017026, 0.0040711, -0.0030879, 0.0032005
3: 0.0014090, 0.0021728, 0.0014020, 0.0021660, -0.0004086, 0.0004235
4: 0.0030111, 0.0073245, 0.0030493, 0.0073643, -0.0023918, 0.0023077
5: 0.9963428, 0.9975411, 0.9963534, 0.9975522, -0.0006645, 0.0006411
6: 0.0045640, 0.0056518, 0.0045737, 0.0056618, -0.0006032, 0.0005820
7: -0.0063493, -0.0022900, -0.0063134, -0.0022525, -0.0022510, 0.0021718
8: -0.0074106, -0.0042512, -0.0074397, -0.0042792, -0.0016903, 0.0017519
9: -0.0036430, -0.0033704, -0.0036405, -0.0033679, -0.0001511, 0.0001458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004590, upper bound: 0.0004601
time: 0.89 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004628, upper bound: 0.0004601
time: 0.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.29 seconds
IS_A1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003923, upper bound: 0.0003930
IS_A1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003923, upper bound: 0.0003934
IS_A1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003825
IS_A1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003841
IS_A1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003940, upper bound: 0.0003920
IS_A1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003940, upper bound: 0.0003920
IS_A1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003902, upper bound: 0.0003802
IS_A1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003903, upper bound: 0.0003816
IS_A1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003923, upper bound: 0.0003968
IS_A1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003924, upper bound: 0.0003972
IS_A1_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003842
IS_A1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003862
IS_A1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003940, upper bound: 0.0003963
IS_A1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003941, upper bound: 0.0003965
IS_A1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003903, upper bound: 0.0003829
IS_A1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003902, upper bound: 0.0003848
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004064, upper bound: 0.0004048
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003967, upper bound: 0.0004019
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004055, upper bound: 0.0004068
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003948, upper bound: 0.0004030
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004063, upper bound: 0.0004099
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003968, upper bound: 0.0004083
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004055, upper bound: 0.0004127
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003950, upper bound: 0.0004107
IS_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004530
IS_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004529
IS_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004600
IS_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004600
IS_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004533
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004533
IS_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004616
IS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004604, upper bound: 0.0004616
IS_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003971, upper bound: 0.0003930
IS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003972, upper bound: 0.0003934
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003958, upper bound: 0.0003826
IS_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003958, upper bound: 0.0003844
IS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004000, upper bound: 0.0003919
IS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004000, upper bound: 0.0003920
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003983, upper bound: 0.0003806
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003983, upper bound: 0.0003821
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004001, upper bound: 0.0003986
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004001, upper bound: 0.0003993
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003986, upper bound: 0.0003892
IS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003986, upper bound: 0.0003910
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004024, upper bound: 0.0003983
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004024, upper bound: 0.0003990
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004009, upper bound: 0.0003878
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004009, upper bound: 0.0003901
IS_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004100, upper bound: 0.0004049
IS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003984, upper bound: 0.0004017
IS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004097, upper bound: 0.0004068
IS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0003974, upper bound: 0.0004030
IS_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004563
IS_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004600
IS_A2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004624, upper bound: 0.0004563
IS_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004625, upper bound: 0.0004600
IS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004134, upper bound: 0.0004101
IS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004034, upper bound: 0.0004082
IS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004128, upper bound: 0.0004123
IS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004025, upper bound: 0.0004101
IS_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004590, upper bound: 0.0004526
IS_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004628, upper bound: 0.0004526
IS_A2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004590, upper bound: 0.0004601
IS_A2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 5, lower bound: -0.0004628, upper bound: 0.0004601

## BFS IS instance: IS_A1_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0089068, -0.0062512, -0.0090431, -0.0063213, -0.0014178, 0.0015618
1: -0.0054498, -0.0047011, -0.0054883, -0.0047209, -0.0003997, 0.0004403
2: -0.0016501, 0.0038742, -0.0019337, 0.0037283, -0.0029494, 0.0032489
3: 0.0014089, 0.0021400, 0.0013714, 0.0021207, -0.0003903, 0.0004299
4: 0.0031966, 0.0073250, 0.0033055, 0.0075370, -0.0024280, 0.0022042
5: 0.9963943, 0.9975414, 0.9964246, 0.9976003, -0.0006746, 0.0006124
6: 0.0046108, 0.0056519, 0.0046383, 0.0057054, -0.0006123, 0.0005559
7: -0.0061748, -0.0022895, -0.0060723, -0.0020900, -0.0022850, 0.0020744
8: -0.0074110, -0.0043870, -0.0075662, -0.0044668, -0.0016145, 0.0017784
9: -0.0036312, -0.0033704, -0.0036244, -0.0033570, -0.0001534, 0.0001393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003868
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003930
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088964, -0.0062357, -0.0090376, -0.0063216, -0.0014225, 0.0015967
1: -0.0054469, -0.0046967, -0.0054867, -0.0047209, -0.0004011, 0.0004502
2: -0.0016284, 0.0039063, -0.0019223, 0.0037277, -0.0029591, 0.0033216
3: 0.0014118, 0.0021442, 0.0013729, 0.0021206, -0.0003916, 0.0004396
4: 0.0031726, 0.0073088, 0.0033060, 0.0075285, -0.0024823, 0.0022115
5: 0.9963877, 0.9975369, 0.9964248, 0.9975979, -0.0006897, 0.0006144
6: 0.0046047, 0.0056478, 0.0046384, 0.0057032, -0.0006260, 0.0005577
7: -0.0061974, -0.0023047, -0.0060718, -0.0020980, -0.0023361, 0.0020812
8: -0.0073991, -0.0043694, -0.0075600, -0.0044672, -0.0016198, 0.0018182
9: -0.0036328, -0.0033714, -0.0036243, -0.0033575, -0.0001569, 0.0001398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003868
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003933
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090359, -0.0063059, -0.0090424, -0.0063529, -0.0015676, 0.0015802
1: -0.0054862, -0.0047165, -0.0054880, -0.0047298, -0.0004420, 0.0004455
2: -0.0019187, 0.0037603, -0.0019321, 0.0036624, -0.0032609, 0.0032871
3: 0.0013734, 0.0021249, 0.0013716, 0.0021120, -0.0004315, 0.0004350
4: 0.0032817, 0.0075258, 0.0033548, 0.0075358, -0.0024566, 0.0024370
5: 0.9964179, 0.9975972, 0.9964383, 0.9975999, -0.0006825, 0.0006771
6: 0.0046323, 0.0057026, 0.0046507, 0.0057051, -0.0006195, 0.0006146
7: -0.0060947, -0.0021005, -0.0060259, -0.0020911, -0.0023119, 0.0022935
8: -0.0075580, -0.0044493, -0.0075654, -0.0045029, -0.0017850, 0.0017993
9: -0.0036259, -0.0033577, -0.0036212, -0.0033570, -0.0001552, 0.0001540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003638, upper bound: 0.0003664
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003804, upper bound: 0.0003746
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0090209, -0.0062940, -0.0090369, -0.0063532, -0.0015679, 0.0016160
1: -0.0054820, -0.0047132, -0.0054865, -0.0047299, -0.0004420, 0.0004556
2: -0.0018874, 0.0037851, -0.0019207, 0.0036618, -0.0032615, 0.0033616
3: 0.0013775, 0.0021282, 0.0013731, 0.0021119, -0.0004316, 0.0004448
4: 0.0032631, 0.0075024, 0.0033553, 0.0075273, -0.0025122, 0.0024374
5: 0.9964128, 0.9975907, 0.9964384, 0.9975976, -0.0006980, 0.0006772
6: 0.0046276, 0.0056967, 0.0046508, 0.0057029, -0.0006335, 0.0006147
7: -0.0061122, -0.0021226, -0.0060255, -0.0020991, -0.0023643, 0.0022939
8: -0.0075409, -0.0044357, -0.0075591, -0.0045032, -0.0017853, 0.0018401
9: -0.0036270, -0.0033591, -0.0036212, -0.0033576, -0.0001588, 0.0001540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003638, upper bound: 0.0003673
time: 1.07 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003804, upper bound: 0.0003760
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0089060, -0.0062513, -0.0090610, -0.0062860, -0.0014503, 0.0015721
1: -0.0054496, -0.0047011, -0.0054933, -0.0047109, -0.0004089, 0.0004432
2: -0.0016484, 0.0038738, -0.0019709, 0.0038016, -0.0030169, 0.0032703
3: 0.0014092, 0.0021399, 0.0013665, 0.0021304, -0.0003992, 0.0004328
4: 0.0031968, 0.0073238, 0.0032508, 0.0075648, -0.0024440, 0.0022547
5: 0.9963944, 0.9975410, 0.9964094, 0.9976079, -0.0006790, 0.0006264
6: 0.0046109, 0.0056516, 0.0046245, 0.0057124, -0.0006163, 0.0005686
7: -0.0061746, -0.0022907, -0.0061238, -0.0020638, -0.0023001, 0.0021219
8: -0.0074100, -0.0043872, -0.0075866, -0.0044267, -0.0016515, 0.0017902
9: -0.0036312, -0.0033704, -0.0036278, -0.0033552, -0.0001544, 0.0001425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003854
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003919
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088962, -0.0062359, -0.0090552, -0.0062863, -0.0014550, 0.0016073
1: -0.0054468, -0.0046968, -0.0054917, -0.0047110, -0.0004102, 0.0004532
2: -0.0016280, 0.0039059, -0.0019589, 0.0038010, -0.0030267, 0.0033435
3: 0.0014119, 0.0021442, 0.0013681, 0.0021303, -0.0004005, 0.0004425
4: 0.0031728, 0.0073085, 0.0032512, 0.0075558, -0.0024987, 0.0022619
5: 0.9963877, 0.9975368, 0.9964095, 0.9976054, -0.0006942, 0.0006284
6: 0.0046048, 0.0056478, 0.0046246, 0.0057101, -0.0006301, 0.0005704
7: -0.0061972, -0.0023050, -0.0061234, -0.0020723, -0.0023516, 0.0021287
8: -0.0073989, -0.0043696, -0.0075800, -0.0044270, -0.0016568, 0.0018302
9: -0.0036327, -0.0033714, -0.0036278, -0.0033558, -0.0001579, 0.0001429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003852
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003920
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090354, -0.0063061, -0.0090602, -0.0063207, -0.0015946, 0.0015884
1: -0.0054861, -0.0047166, -0.0054931, -0.0047207, -0.0004496, 0.0004478
2: -0.0019176, 0.0037599, -0.0019693, 0.0037295, -0.0033170, 0.0033042
3: 0.0013735, 0.0021249, 0.0013667, 0.0021208, -0.0004390, 0.0004373
4: 0.0032819, 0.0075250, 0.0033047, 0.0075636, -0.0024693, 0.0024789
5: 0.9964181, 0.9975970, 0.9964244, 0.9976076, -0.0006861, 0.0006887
6: 0.0046323, 0.0057024, 0.0046381, 0.0057121, -0.0006227, 0.0006251
7: -0.0060945, -0.0021013, -0.0060731, -0.0020650, -0.0023239, 0.0023329
8: -0.0075574, -0.0044495, -0.0075857, -0.0044662, -0.0018157, 0.0018087
9: -0.0036259, -0.0033577, -0.0036244, -0.0033553, -0.0001560, 0.0001567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003641, upper bound: 0.0003600
time: 1.08 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003815, upper bound: 0.0003722
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0090201, -0.0062941, -0.0090544, -0.0063210, -0.0015950, 0.0016245
1: -0.0054818, -0.0047132, -0.0054914, -0.0047208, -0.0004497, 0.0004580
2: -0.0018858, 0.0037848, -0.0019572, 0.0037289, -0.0033180, 0.0033792
3: 0.0013777, 0.0021282, 0.0013683, 0.0021208, -0.0004391, 0.0004472
4: 0.0032633, 0.0075012, 0.0033051, 0.0075546, -0.0025254, 0.0024796
5: 0.9964129, 0.9975903, 0.9964244, 0.9976051, -0.0007016, 0.0006889
6: 0.0046276, 0.0056964, 0.0046382, 0.0057098, -0.0006369, 0.0006253
7: -0.0061120, -0.0021237, -0.0060727, -0.0020734, -0.0023767, 0.0023336
8: -0.0075400, -0.0044359, -0.0075791, -0.0044665, -0.0018163, 0.0018498
9: -0.0036270, -0.0033592, -0.0036244, -0.0033558, -0.0001596, 0.0001567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003641, upper bound: 0.0003615
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003816, upper bound: 0.0003732
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0089068, -0.0062512, -0.0090142, -0.0062704, -0.0015566, 0.0016330
1: -0.0054498, -0.0047011, -0.0054801, -0.0047065, -0.0004389, 0.0004604
2: -0.0016501, 0.0038742, -0.0018736, 0.0038341, -0.0032381, 0.0033969
3: 0.0014089, 0.0021400, 0.0013794, 0.0021347, -0.0004285, 0.0004495
4: 0.0031966, 0.0073250, 0.0032265, 0.0074921, -0.0025386, 0.0024200
5: 0.9963943, 0.9975414, 0.9964027, 0.9975877, -0.0007053, 0.0006723
6: 0.0046108, 0.0056519, 0.0046183, 0.0056941, -0.0006402, 0.0006103
7: -0.0061748, -0.0022895, -0.0061467, -0.0021323, -0.0023891, 0.0022775
8: -0.0074110, -0.0043870, -0.0075333, -0.0044089, -0.0017726, 0.0018595
9: -0.0036312, -0.0033704, -0.0036294, -0.0033598, -0.0001604, 0.0001529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003922
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003968
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088964, -0.0062357, -0.0090080, -0.0062707, -0.0015615, 0.0016593
1: -0.0054469, -0.0046967, -0.0054783, -0.0047066, -0.0004403, 0.0004678
2: -0.0016284, 0.0039063, -0.0018606, 0.0038334, -0.0032483, 0.0034518
3: 0.0014118, 0.0021442, 0.0013811, 0.0021346, -0.0004299, 0.0004568
4: 0.0031726, 0.0073088, 0.0032270, 0.0074824, -0.0025796, 0.0024276
5: 0.9963877, 0.9975369, 0.9964028, 0.9975851, -0.0007167, 0.0006745
6: 0.0046047, 0.0056478, 0.0046185, 0.0056916, -0.0006505, 0.0006122
7: -0.0061974, -0.0023047, -0.0061462, -0.0021414, -0.0024277, 0.0022846
8: -0.0073991, -0.0043694, -0.0075262, -0.0044093, -0.0017781, 0.0018895
9: -0.0036328, -0.0033714, -0.0036293, -0.0033604, -0.0001630, 0.0001534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003925
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003972
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090359, -0.0063059, -0.0090135, -0.0063022, -0.0016988, 0.0016514
1: -0.0054862, -0.0047165, -0.0054799, -0.0047155, -0.0004790, 0.0004656
2: -0.0019187, 0.0037603, -0.0018721, 0.0037680, -0.0035339, 0.0034352
3: 0.0013734, 0.0021249, 0.0013795, 0.0021259, -0.0004677, 0.0004546
4: 0.0032817, 0.0075258, 0.0032759, 0.0074910, -0.0025673, 0.0026410
5: 0.9964179, 0.9975972, 0.9964164, 0.9975874, -0.0007133, 0.0007337
6: 0.0046323, 0.0057026, 0.0046308, 0.0056938, -0.0006474, 0.0006660
7: -0.0060947, -0.0021005, -0.0061001, -0.0021333, -0.0024161, 0.0024855
8: -0.0075580, -0.0044493, -0.0075325, -0.0044451, -0.0019344, 0.0018805
9: -0.0036259, -0.0033577, -0.0036262, -0.0033599, -0.0001622, 0.0001669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003638, upper bound: 0.0003677
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003804, upper bound: 0.0003758
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0090209, -0.0062940, -0.0090073, -0.0063025, -0.0016993, 0.0016786
1: -0.0054820, -0.0047132, -0.0054781, -0.0047156, -0.0004791, 0.0004733
2: -0.0018874, 0.0037851, -0.0018591, 0.0037673, -0.0035349, 0.0034918
3: 0.0013775, 0.0021282, 0.0013813, 0.0021258, -0.0004678, 0.0004621
4: 0.0032631, 0.0075024, 0.0032764, 0.0074813, -0.0026096, 0.0026418
5: 0.9964128, 0.9975907, 0.9964165, 0.9975848, -0.0007250, 0.0007340
6: 0.0046276, 0.0056967, 0.0046309, 0.0056913, -0.0006581, 0.0006662
7: -0.0061122, -0.0021226, -0.0060997, -0.0021424, -0.0024559, 0.0024862
8: -0.0075409, -0.0044357, -0.0075254, -0.0044455, -0.0019350, 0.0019114
9: -0.0036270, -0.0033591, -0.0036262, -0.0033605, -0.0001649, 0.0001669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003639, upper bound: 0.0003688
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003804, upper bound: 0.0003777
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0089060, -0.0062513, -0.0090324, -0.0062365, -0.0015970, 0.0016452
1: -0.0054496, -0.0047011, -0.0054852, -0.0046970, -0.0004503, 0.0004638
2: -0.0016484, 0.0038738, -0.0019113, 0.0039047, -0.0033221, 0.0034223
3: 0.0014092, 0.0021399, 0.0013744, 0.0021440, -0.0004396, 0.0004529
4: 0.0031968, 0.0073238, 0.0031738, 0.0075203, -0.0025576, 0.0024828
5: 0.9963944, 0.9975410, 0.9963880, 0.9975956, -0.0007106, 0.0006898
6: 0.0046109, 0.0056516, 0.0046050, 0.0057012, -0.0006450, 0.0006261
7: -0.0061746, -0.0022907, -0.0061963, -0.0021057, -0.0024070, 0.0023365
8: -0.0074100, -0.0043872, -0.0075540, -0.0043703, -0.0018185, 0.0018734
9: -0.0036312, -0.0033704, -0.0036327, -0.0033580, -0.0001616, 0.0001569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003910
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003962
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088962, -0.0062359, -0.0090250, -0.0062368, -0.0016018, 0.0016710
1: -0.0054468, -0.0046968, -0.0054831, -0.0046971, -0.0004516, 0.0004711
2: -0.0016280, 0.0039059, -0.0018960, 0.0039040, -0.0033322, 0.0034760
3: 0.0014119, 0.0021442, 0.0013764, 0.0021439, -0.0004410, 0.0004600
4: 0.0031728, 0.0073085, 0.0031743, 0.0075088, -0.0025977, 0.0024902
5: 0.9963877, 0.9975368, 0.9963882, 0.9975923, -0.0007217, 0.0006919
6: 0.0046048, 0.0056478, 0.0046052, 0.0056983, -0.0006551, 0.0006280
7: -0.0061972, -0.0023050, -0.0061958, -0.0021165, -0.0024448, 0.0023436
8: -0.0073989, -0.0043696, -0.0075456, -0.0043707, -0.0018240, 0.0019028
9: -0.0036327, -0.0033714, -0.0036327, -0.0033587, -0.0001642, 0.0001574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003913
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003964
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090354, -0.0063061, -0.0090316, -0.0062716, -0.0017364, 0.0016616
1: -0.0054861, -0.0047166, -0.0054850, -0.0047069, -0.0004896, 0.0004685
2: -0.0019176, 0.0037599, -0.0019097, 0.0038316, -0.0036121, 0.0034564
3: 0.0013735, 0.0021249, 0.0013746, 0.0021344, -0.0004780, 0.0004574
4: 0.0032819, 0.0075250, 0.0032283, 0.0075191, -0.0025831, 0.0026995
5: 0.9964181, 0.9975970, 0.9964032, 0.9975953, -0.0007177, 0.0007500
6: 0.0046323, 0.0057024, 0.0046188, 0.0057009, -0.0006514, 0.0006808
7: -0.0060945, -0.0021013, -0.0061449, -0.0021069, -0.0024310, 0.0025405
8: -0.0075574, -0.0044495, -0.0075531, -0.0044103, -0.0019773, 0.0018920
9: -0.0036259, -0.0033577, -0.0036292, -0.0033581, -0.0001632, 0.0001706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003641, upper bound: 0.0003625
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003816, upper bound: 0.0003743
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0090201, -0.0062941, -0.0090243, -0.0062719, -0.0017369, 0.0016882
1: -0.0054818, -0.0047132, -0.0054829, -0.0047069, -0.0004897, 0.0004760
2: -0.0018858, 0.0037848, -0.0018945, 0.0038310, -0.0036132, 0.0035118
3: 0.0013777, 0.0021282, 0.0013766, 0.0021343, -0.0004782, 0.0004647
4: 0.0032633, 0.0075012, 0.0032288, 0.0075077, -0.0026245, 0.0027003
5: 0.9964129, 0.9975903, 0.9964033, 0.9975921, -0.0007292, 0.0007502
6: 0.0046276, 0.0056964, 0.0046189, 0.0056980, -0.0006619, 0.0006810
7: -0.0061120, -0.0021237, -0.0061444, -0.0021176, -0.0024699, 0.0025413
8: -0.0075400, -0.0044359, -0.0075447, -0.0044106, -0.0019779, 0.0019223
9: -0.0036270, -0.0033592, -0.0036292, -0.0033588, -0.0001659, 0.0001706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003641, upper bound: 0.0003635
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003816, upper bound: 0.0003762
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090544, -0.0063208, -0.0089576, -0.0062477, -0.0016424, 0.0014775
1: -0.0054914, -0.0047207, -0.0054641, -0.0047001, -0.0004631, 0.0004166
2: -0.0019572, 0.0037294, -0.0017558, 0.0038815, -0.0034166, 0.0030734
3: 0.0013683, 0.0021208, 0.0013949, 0.0021409, -0.0004521, 0.0004067
4: 0.0033048, 0.0075546, 0.0031911, 0.0074040, -0.0022969, 0.0025533
5: 0.9964244, 0.9976051, 0.9963928, 0.9975633, -0.0006381, 0.0007094
6: 0.0046381, 0.0057098, 0.0046094, 0.0056719, -0.0005792, 0.0006439
7: -0.0060730, -0.0020734, -0.0061800, -0.0022151, -0.0021616, 0.0024030
8: -0.0075791, -0.0044662, -0.0074688, -0.0043830, -0.0018702, 0.0016824
9: -0.0036244, -0.0033558, -0.0036316, -0.0033654, -0.0001451, 0.0001614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003929, upper bound: 0.0003923
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003934, upper bound: 0.0003923
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090537, -0.0063525, -0.0090893, -0.0063001, -0.0016603, 0.0016256
1: -0.0054912, -0.0047297, -0.0055013, -0.0047149, -0.0004681, 0.0004583
2: -0.0019556, 0.0036634, -0.0020298, 0.0037724, -0.0034538, 0.0033815
3: 0.0013685, 0.0021121, 0.0013587, 0.0021265, -0.0004571, 0.0004475
4: 0.0033540, 0.0075534, 0.0032726, 0.0076088, -0.0025272, 0.0025811
5: 0.9964381, 0.9976048, 0.9964154, 0.9976202, -0.0007021, 0.0007171
6: 0.0046505, 0.0057095, 0.0046300, 0.0057235, -0.0006373, 0.0006509
7: -0.0060266, -0.0020746, -0.0061032, -0.0020224, -0.0023783, 0.0024291
8: -0.0075782, -0.0045023, -0.0076188, -0.0044427, -0.0018906, 0.0018511
9: -0.0036213, -0.0033559, -0.0036264, -0.0033524, -0.0001597, 0.0001631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003825, upper bound: 0.0003895
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003841, upper bound: 0.0003895
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090735, -0.0062855, -0.0089569, -0.0062478, -0.0016540, 0.0015108
1: -0.0054968, -0.0047108, -0.0054640, -0.0047002, -0.0004663, 0.0004260
2: -0.0019969, 0.0038027, -0.0017544, 0.0038811, -0.0034408, 0.0031428
3: 0.0013630, 0.0021305, 0.0013951, 0.0021409, -0.0004553, 0.0004159
4: 0.0032500, 0.0075842, 0.0031914, 0.0074030, -0.0023488, 0.0025714
5: 0.9964092, 0.9976133, 0.9963929, 0.9975630, -0.0006526, 0.0007144
6: 0.0046243, 0.0057173, 0.0046095, 0.0056716, -0.0005923, 0.0006485
7: -0.0061245, -0.0020455, -0.0061797, -0.0022161, -0.0022104, 0.0024200
8: -0.0076008, -0.0044261, -0.0074681, -0.0043832, -0.0018835, 0.0017204
9: -0.0036279, -0.0033540, -0.0036316, -0.0033654, -0.0001484, 0.0001625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003920, upper bound: 0.0003941
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003920, upper bound: 0.0003941
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090727, -0.0063202, -0.0090889, -0.0063002, -0.0016699, 0.0016526
1: -0.0054966, -0.0047206, -0.0055012, -0.0047149, -0.0004708, 0.0004659
2: -0.0019952, 0.0037306, -0.0020289, 0.0037721, -0.0034738, 0.0034377
3: 0.0013633, 0.0021210, 0.0013588, 0.0021265, -0.0004597, 0.0004549
4: 0.0033039, 0.0075829, 0.0032729, 0.0076082, -0.0025691, 0.0025961
5: 0.9964241, 0.9976130, 0.9964155, 0.9976200, -0.0007138, 0.0007213
6: 0.0046379, 0.0057170, 0.0046300, 0.0057233, -0.0006479, 0.0006547
7: -0.0060738, -0.0020467, -0.0061030, -0.0020230, -0.0024178, 0.0024432
8: -0.0075999, -0.0044656, -0.0076184, -0.0044429, -0.0019016, 0.0018818
9: -0.0036245, -0.0033541, -0.0036264, -0.0033525, -0.0001624, 0.0001641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003903
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003815, upper bound: 0.0003903
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090544, -0.0063208, -0.0089312, -0.0061949, -0.0017735, 0.0015464
1: -0.0054914, -0.0047207, -0.0054567, -0.0046852, -0.0005000, 0.0004360
2: -0.0019572, 0.0037294, -0.0017009, 0.0039912, -0.0036893, 0.0032167
3: 0.0013683, 0.0021208, 0.0014022, 0.0021555, -0.0004882, 0.0004257
4: 0.0033048, 0.0075546, 0.0031091, 0.0073630, -0.0024040, 0.0027571
5: 0.9964244, 0.9976051, 0.9963701, 0.9975520, -0.0006679, 0.0007660
6: 0.0046381, 0.0057098, 0.0045887, 0.0056615, -0.0006062, 0.0006953
7: -0.0060730, -0.0020734, -0.0062571, -0.0022537, -0.0022624, 0.0025948
8: -0.0075791, -0.0044662, -0.0074388, -0.0043229, -0.0020195, 0.0017608
9: -0.0036244, -0.0033558, -0.0036368, -0.0033680, -0.0001519, 0.0001742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003929, upper bound: 0.0003972
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003934, upper bound: 0.0003972
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090537, -0.0063525, -0.0090657, -0.0062452, -0.0017910, 0.0016779
1: -0.0054912, -0.0047297, -0.0054946, -0.0046994, -0.0005049, 0.0004731
2: -0.0019556, 0.0036634, -0.0019807, 0.0038865, -0.0037256, 0.0034903
3: 0.0013685, 0.0021121, 0.0013652, 0.0021416, -0.0004930, 0.0004619
4: 0.0033540, 0.0075534, 0.0031873, 0.0075721, -0.0026085, 0.0027843
5: 0.9964381, 0.9976048, 0.9963918, 0.9976100, -0.0007247, 0.0007736
6: 0.0046505, 0.0057095, 0.0046085, 0.0057143, -0.0006578, 0.0007022
7: -0.0060266, -0.0020746, -0.0061835, -0.0020569, -0.0024549, 0.0026203
8: -0.0075782, -0.0045023, -0.0075920, -0.0043802, -0.0020394, 0.0019106
9: -0.0036213, -0.0033559, -0.0036318, -0.0033547, -0.0001648, 0.0001760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003824, upper bound: 0.0003958
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003958
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090735, -0.0062855, -0.0089307, -0.0061951, -0.0017851, 0.0015776
1: -0.0054968, -0.0047108, -0.0054566, -0.0046853, -0.0005033, 0.0004448
2: -0.0019969, 0.0038027, -0.0016998, 0.0039908, -0.0037134, 0.0032817
3: 0.0013630, 0.0021305, 0.0014023, 0.0021554, -0.0004914, 0.0004343
4: 0.0032500, 0.0075842, 0.0031094, 0.0073622, -0.0024525, 0.0027751
5: 0.9964092, 0.9976133, 0.9963702, 0.9975517, -0.0006814, 0.0007710
6: 0.0046243, 0.0057173, 0.0045888, 0.0056613, -0.0006185, 0.0006998
7: -0.0061245, -0.0020455, -0.0062568, -0.0022545, -0.0023081, 0.0026117
8: -0.0076008, -0.0044261, -0.0074382, -0.0043232, -0.0020327, 0.0017964
9: -0.0036279, -0.0033540, -0.0036368, -0.0033680, -0.0001550, 0.0001754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003919, upper bound: 0.0004000
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003921, upper bound: 0.0004000
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090727, -0.0063202, -0.0090639, -0.0062454, -0.0018005, 0.0017055
1: -0.0054966, -0.0047206, -0.0054941, -0.0046995, -0.0005076, 0.0004808
2: -0.0019952, 0.0037306, -0.0019769, 0.0038861, -0.0037455, 0.0035478
3: 0.0013633, 0.0021210, 0.0013657, 0.0021416, -0.0004957, 0.0004695
4: 0.0033039, 0.0075829, 0.0031876, 0.0075693, -0.0026514, 0.0027992
5: 0.9964241, 0.9976130, 0.9963918, 0.9976092, -0.0007366, 0.0007777
6: 0.0046379, 0.0057170, 0.0046085, 0.0057135, -0.0006686, 0.0007059
7: -0.0060738, -0.0020467, -0.0061832, -0.0020596, -0.0024953, 0.0026343
8: -0.0075999, -0.0044656, -0.0075899, -0.0043804, -0.0020503, 0.0019421
9: -0.0036245, -0.0033541, -0.0036318, -0.0033549, -0.0001676, 0.0001769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003806, upper bound: 0.0003983
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003820, upper bound: 0.0003983
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0089212, -0.0062483, -0.0089589, -0.0062095, -0.0014581, 0.0014638
1: -0.0054539, -0.0047003, -0.0054645, -0.0046893, -0.0004111, 0.0004127
2: -0.0016801, 0.0038800, -0.0017584, 0.0039609, -0.0030331, 0.0030451
3: 0.0014050, 0.0021408, 0.0013946, 0.0021515, -0.0004014, 0.0004030
4: 0.0031922, 0.0073475, 0.0031317, 0.0074060, -0.0022757, 0.0022667
5: 0.9963931, 0.9975476, 0.9963763, 0.9975638, -0.0006323, 0.0006298
6: 0.0046097, 0.0056576, 0.0045944, 0.0056724, -0.0005739, 0.0005716
7: -0.0061789, -0.0022683, -0.0062358, -0.0022133, -0.0021417, 0.0021332
8: -0.0074274, -0.0043838, -0.0074703, -0.0043395, -0.0016603, 0.0016669
9: -0.0036315, -0.0033689, -0.0036353, -0.0033652, -0.0001438, 0.0001432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004387
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004406
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0090529, -0.0063009, -0.0089579, -0.0062388, -0.0015923, 0.0014848
1: -0.0054910, -0.0047151, -0.0054642, -0.0046976, -0.0004489, 0.0004186
2: -0.0019541, 0.0037708, -0.0017565, 0.0038999, -0.0033124, 0.0030887
3: 0.0013687, 0.0021263, 0.0013949, 0.0021434, -0.0004383, 0.0004087
4: 0.0032738, 0.0075522, 0.0031773, 0.0074045, -0.0023083, 0.0024755
5: 0.9964157, 0.9976045, 0.9963890, 0.9975635, -0.0006413, 0.0006878
6: 0.0046303, 0.0057092, 0.0046059, 0.0056720, -0.0005821, 0.0006243
7: -0.0061021, -0.0020756, -0.0061929, -0.0022146, -0.0021724, 0.0023297
8: -0.0075774, -0.0044436, -0.0074692, -0.0043729, -0.0018132, 0.0016907
9: -0.0036264, -0.0033560, -0.0036325, -0.0033653, -0.0001459, 0.0001564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004387
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004405
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0089296, -0.0062241, -0.0089582, -0.0062096, -0.0014666, 0.0014975
1: -0.0054562, -0.0046935, -0.0054643, -0.0046894, -0.0004135, 0.0004222
2: -0.0016974, 0.0039305, -0.0017571, 0.0039606, -0.0030508, 0.0031150
3: 0.0014027, 0.0021474, 0.0013948, 0.0021514, -0.0004037, 0.0004122
4: 0.0031545, 0.0073604, 0.0031320, 0.0074050, -0.0023280, 0.0022800
5: 0.9963827, 0.9975511, 0.9963763, 0.9975636, -0.0006468, 0.0006334
6: 0.0046002, 0.0056609, 0.0045945, 0.0056721, -0.0005871, 0.0005750
7: -0.0062144, -0.0022562, -0.0062356, -0.0022142, -0.0021909, 0.0021457
8: -0.0074369, -0.0043562, -0.0074695, -0.0043397, -0.0016700, 0.0017052
9: -0.0036339, -0.0033681, -0.0036353, -0.0033653, -0.0001471, 0.0001441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004461
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004474
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0090611, -0.0062784, -0.0089573, -0.0062389, -0.0016003, 0.0015194
1: -0.0054933, -0.0047088, -0.0054640, -0.0046976, -0.0004512, 0.0004284
2: -0.0019710, 0.0038176, -0.0017551, 0.0038996, -0.0033289, 0.0031606
3: 0.0013665, 0.0021325, 0.0013950, 0.0021433, -0.0004405, 0.0004183
4: 0.0032388, 0.0075649, 0.0031776, 0.0074035, -0.0023620, 0.0024878
5: 0.9964061, 0.9976080, 0.9963891, 0.9975632, -0.0006562, 0.0006912
6: 0.0046215, 0.0057124, 0.0046060, 0.0056717, -0.0005957, 0.0006274
7: -0.0061350, -0.0020638, -0.0061927, -0.0022156, -0.0022229, 0.0023413
8: -0.0075866, -0.0044180, -0.0074684, -0.0043731, -0.0018222, 0.0017301
9: -0.0036286, -0.0033552, -0.0036324, -0.0033654, -0.0001493, 0.0001572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004461
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004474
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0089212, -0.0062483, -0.0089325, -0.0061563, -0.0015957, 0.0015346
1: -0.0054539, -0.0047003, -0.0054571, -0.0046743, -0.0004499, 0.0004327
2: -0.0016801, 0.0038800, -0.0017036, 0.0040715, -0.0033194, 0.0031922
3: 0.0014050, 0.0021408, 0.0014019, 0.0021661, -0.0004393, 0.0004224
4: 0.0031922, 0.0073475, 0.0030491, 0.0073650, -0.0023857, 0.0024807
5: 0.9963931, 0.9975476, 0.9963534, 0.9975525, -0.0006628, 0.0006892
6: 0.0046097, 0.0056576, 0.0045736, 0.0056620, -0.0006016, 0.0006256
7: -0.0061789, -0.0022683, -0.0063136, -0.0022518, -0.0022452, 0.0023346
8: -0.0074274, -0.0043838, -0.0074403, -0.0042790, -0.0018170, 0.0017474
9: -0.0036315, -0.0033689, -0.0036406, -0.0033678, -0.0001508, 0.0001568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004423, upper bound: 0.0004401
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004401
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0090529, -0.0063009, -0.0089316, -0.0061853, -0.0017234, 0.0015556
1: -0.0054910, -0.0047151, -0.0054568, -0.0046825, -0.0004859, 0.0004386
2: -0.0019541, 0.0037708, -0.0017017, 0.0040111, -0.0035850, 0.0032360
3: 0.0013687, 0.0021263, 0.0014021, 0.0021581, -0.0004744, 0.0004282
4: 0.0032738, 0.0075522, 0.0030942, 0.0073636, -0.0024184, 0.0026792
5: 0.9964157, 0.9976045, 0.9963659, 0.9975521, -0.0006719, 0.0007444
6: 0.0046303, 0.0057092, 0.0045850, 0.0056617, -0.0006099, 0.0006757
7: -0.0061021, -0.0020756, -0.0062711, -0.0022532, -0.0022760, 0.0025214
8: -0.0075774, -0.0044436, -0.0074392, -0.0043120, -0.0019624, 0.0017714
9: -0.0036264, -0.0033560, -0.0036377, -0.0033679, -0.0001528, 0.0001693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004401
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004401
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0089296, -0.0062241, -0.0089320, -0.0061565, -0.0016042, 0.0015717
1: -0.0054562, -0.0046935, -0.0054569, -0.0046744, -0.0004523, 0.0004431
2: -0.0016974, 0.0039305, -0.0017026, 0.0040711, -0.0033370, 0.0032695
3: 0.0014027, 0.0021474, 0.0014020, 0.0021660, -0.0004416, 0.0004327
4: 0.0031545, 0.0073604, 0.0030493, 0.0073643, -0.0024435, 0.0024939
5: 0.9963827, 0.9975511, 0.9963534, 0.9975522, -0.0006789, 0.0006929
6: 0.0046002, 0.0056609, 0.0045737, 0.0056618, -0.0006162, 0.0006289
7: -0.0062144, -0.0022562, -0.0063134, -0.0022525, -0.0022996, 0.0023470
8: -0.0074369, -0.0043562, -0.0074397, -0.0042792, -0.0018267, 0.0017898
9: -0.0036339, -0.0033681, -0.0036405, -0.0033679, -0.0001544, 0.0001576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004423, upper bound: 0.0004486
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004486
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0090611, -0.0062784, -0.0089311, -0.0061855, -0.0017313, 0.0015937
1: -0.0054933, -0.0047088, -0.0054567, -0.0046826, -0.0004881, 0.0004493
2: -0.0019710, 0.0038176, -0.0017006, 0.0040107, -0.0036014, 0.0033152
3: 0.0013665, 0.0021325, 0.0014022, 0.0021580, -0.0004766, 0.0004387
4: 0.0032388, 0.0075649, 0.0030945, 0.0073628, -0.0024776, 0.0026915
5: 0.9964061, 0.9976080, 0.9963660, 0.9975518, -0.0006883, 0.0007478
6: 0.0046215, 0.0057124, 0.0045851, 0.0056615, -0.0006248, 0.0006788
7: -0.0061350, -0.0020638, -0.0062709, -0.0022539, -0.0023317, 0.0025330
8: -0.0075866, -0.0044180, -0.0074386, -0.0043122, -0.0019714, 0.0018147
9: -0.0036286, -0.0033552, -0.0036377, -0.0033680, -0.0001566, 0.0001701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004486
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004486
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0088820, -0.0061979, -0.0090431, -0.0063213, -0.0014911, 0.0016965
1: -0.0054428, -0.0046861, -0.0054883, -0.0047209, -0.0004204, 0.0004783
2: -0.0015985, 0.0039850, -0.0019337, 0.0037283, -0.0031018, 0.0035291
3: 0.0014158, 0.0021546, 0.0013714, 0.0021207, -0.0004105, 0.0004670
4: 0.0031138, 0.0072865, 0.0033055, 0.0075370, -0.0026374, 0.0023181
5: 0.9963714, 0.9975307, 0.9964246, 0.9976003, -0.0007327, 0.0006440
6: 0.0045899, 0.0056422, 0.0046383, 0.0057054, -0.0006651, 0.0005846
7: -0.0062527, -0.0023258, -0.0060723, -0.0020900, -0.0024821, 0.0021815
8: -0.0073827, -0.0043263, -0.0075662, -0.0044668, -0.0016979, 0.0019318
9: -0.0036365, -0.0033728, -0.0036244, -0.0033570, -0.0001667, 0.0001465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003868
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003930
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088668, -0.0061883, -0.0090376, -0.0063216, -0.0014957, 0.0017188
1: -0.0054385, -0.0046834, -0.0054867, -0.0047209, -0.0004217, 0.0004846
2: -0.0015669, 0.0040049, -0.0019223, 0.0037277, -0.0031113, 0.0035754
3: 0.0014199, 0.0021573, 0.0013729, 0.0021206, -0.0004117, 0.0004731
4: 0.0030989, 0.0072629, 0.0033060, 0.0075285, -0.0026720, 0.0023252
5: 0.9963672, 0.9975241, 0.9964248, 0.9975979, -0.0007424, 0.0006460
6: 0.0045862, 0.0056363, 0.0046384, 0.0057032, -0.0006738, 0.0005864
7: -0.0062668, -0.0023479, -0.0060718, -0.0020980, -0.0025147, 0.0021882
8: -0.0073655, -0.0043154, -0.0075600, -0.0044672, -0.0017031, 0.0019572
9: -0.0036374, -0.0033743, -0.0036243, -0.0033575, -0.0001689, 0.0001469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003868
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003934
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090149, -0.0062511, -0.0090424, -0.0063529, -0.0016211, 0.0017150
1: -0.0054803, -0.0047011, -0.0054880, -0.0047298, -0.0004570, 0.0004835
2: -0.0018749, 0.0038742, -0.0019321, 0.0036624, -0.0033722, 0.0035675
3: 0.0013792, 0.0021400, 0.0013716, 0.0021120, -0.0004463, 0.0004721
4: 0.0031965, 0.0074930, 0.0033548, 0.0075358, -0.0026661, 0.0025202
5: 0.9963944, 0.9975880, 0.9964383, 0.9975999, -0.0007407, 0.0007002
6: 0.0046108, 0.0056943, 0.0046507, 0.0057051, -0.0006724, 0.0006356
7: -0.0061749, -0.0021314, -0.0060259, -0.0020911, -0.0025091, 0.0023718
8: -0.0075340, -0.0043870, -0.0075654, -0.0045029, -0.0018460, 0.0019529
9: -0.0036313, -0.0033597, -0.0036212, -0.0033570, -0.0001685, 0.0001593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003702, upper bound: 0.0003664
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003866, upper bound: 0.0003745
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0089949, -0.0062418, -0.0090369, -0.0063532, -0.0016222, 0.0017372
1: -0.0054746, -0.0046984, -0.0054865, -0.0047299, -0.0004574, 0.0004898
2: -0.0018333, 0.0038937, -0.0019207, 0.0036618, -0.0033744, 0.0036138
3: 0.0013847, 0.0021426, 0.0013731, 0.0021119, -0.0004466, 0.0004782
4: 0.0031819, 0.0074620, 0.0033553, 0.0075273, -0.0027007, 0.0025218
5: 0.9963902, 0.9975794, 0.9964384, 0.9975976, -0.0007503, 0.0007006
6: 0.0046071, 0.0056865, 0.0046508, 0.0057029, -0.0006811, 0.0006360
7: -0.0061886, -0.0021606, -0.0060255, -0.0020991, -0.0025417, 0.0023733
8: -0.0075113, -0.0043763, -0.0075591, -0.0045032, -0.0018472, 0.0019782
9: -0.0036322, -0.0033617, -0.0036212, -0.0033576, -0.0001707, 0.0001594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003702, upper bound: 0.0003675
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003866, upper bound: 0.0003763
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0088824, -0.0061981, -0.0090610, -0.0062860, -0.0015220, 0.0017068
1: -0.0054429, -0.0046861, -0.0054933, -0.0047109, -0.0004291, 0.0004812
2: -0.0015994, 0.0039846, -0.0019709, 0.0038016, -0.0031662, 0.0035504
3: 0.0014156, 0.0021546, 0.0013665, 0.0021304, -0.0004190, 0.0004698
4: 0.0031140, 0.0072871, 0.0032508, 0.0075648, -0.0026534, 0.0023662
5: 0.9963714, 0.9975308, 0.9964094, 0.9976079, -0.0007372, 0.0006574
6: 0.0045900, 0.0056424, 0.0046245, 0.0057124, -0.0006691, 0.0005967
7: -0.0062525, -0.0023251, -0.0061238, -0.0020638, -0.0024971, 0.0022268
8: -0.0073832, -0.0043265, -0.0075866, -0.0044267, -0.0017332, 0.0019435
9: -0.0036365, -0.0033727, -0.0036278, -0.0033552, -0.0001677, 0.0001495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003854
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003920
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088674, -0.0061885, -0.0090552, -0.0062863, -0.0015265, 0.0017293
1: -0.0054387, -0.0046834, -0.0054917, -0.0047110, -0.0004304, 0.0004876
2: -0.0015681, 0.0040045, -0.0019589, 0.0038010, -0.0031754, 0.0035973
3: 0.0014198, 0.0021572, 0.0013681, 0.0021303, -0.0004202, 0.0004760
4: 0.0030992, 0.0072638, 0.0032512, 0.0075558, -0.0026884, 0.0023731
5: 0.9963673, 0.9975243, 0.9964095, 0.9976054, -0.0007469, 0.0006593
6: 0.0045862, 0.0056365, 0.0046246, 0.0057101, -0.0006780, 0.0005985
7: -0.0062665, -0.0023471, -0.0061234, -0.0020723, -0.0025301, 0.0022334
8: -0.0073661, -0.0043157, -0.0075800, -0.0044270, -0.0017382, 0.0019692
9: -0.0036374, -0.0033742, -0.0036278, -0.0033558, -0.0001699, 0.0001500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003843, upper bound: 0.0003852
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003843, upper bound: 0.0003921
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090134, -0.0062513, -0.0090602, -0.0063207, -0.0016490, 0.0017232
1: -0.0054799, -0.0047011, -0.0054931, -0.0047207, -0.0004649, 0.0004858
2: -0.0018719, 0.0038739, -0.0019693, 0.0037295, -0.0034303, 0.0035845
3: 0.0013796, 0.0021399, 0.0013667, 0.0021208, -0.0004539, 0.0004744
4: 0.0031968, 0.0074908, 0.0033047, 0.0075636, -0.0026789, 0.0025636
5: 0.9963944, 0.9975874, 0.9964244, 0.9976076, -0.0007443, 0.0007122
6: 0.0046109, 0.0056937, 0.0046381, 0.0057121, -0.0006756, 0.0006465
7: -0.0061746, -0.0021334, -0.0060731, -0.0020650, -0.0025211, 0.0024126
8: -0.0075324, -0.0043872, -0.0075857, -0.0044662, -0.0018778, 0.0019622
9: -0.0036312, -0.0033599, -0.0036244, -0.0033553, -0.0001693, 0.0001620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003714, upper bound: 0.0003602
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003725
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0089933, -0.0062420, -0.0090544, -0.0063210, -0.0016502, 0.0017457
1: -0.0054742, -0.0046985, -0.0054914, -0.0047208, -0.0004652, 0.0004922
2: -0.0018300, 0.0038933, -0.0019572, 0.0037289, -0.0034327, 0.0036313
3: 0.0013851, 0.0021425, 0.0013683, 0.0021208, -0.0004543, 0.0004805
4: 0.0031822, 0.0074595, 0.0033051, 0.0075546, -0.0027138, 0.0025654
5: 0.9963904, 0.9975787, 0.9964244, 0.9976051, -0.0007540, 0.0007127
6: 0.0046072, 0.0056858, 0.0046382, 0.0057098, -0.0006844, 0.0006470
7: -0.0061883, -0.0021629, -0.0060727, -0.0020734, -0.0025540, 0.0024143
8: -0.0075094, -0.0043765, -0.0075791, -0.0044665, -0.0018791, 0.0019878
9: -0.0036322, -0.0033619, -0.0036244, -0.0033558, -0.0001715, 0.0001621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003714, upper bound: 0.0003617
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003738
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0088820, -0.0061979, -0.0090142, -0.0062704, -0.0014278, 0.0015651
1: -0.0054428, -0.0046861, -0.0054801, -0.0047065, -0.0004026, 0.0004413
2: -0.0015985, 0.0039850, -0.0018736, 0.0038341, -0.0029701, 0.0032557
3: 0.0014158, 0.0021546, 0.0013794, 0.0021347, -0.0003930, 0.0004308
4: 0.0031138, 0.0072865, 0.0032265, 0.0074921, -0.0024331, 0.0022197
5: 0.9963714, 0.9975307, 0.9964027, 0.9975877, -0.0006760, 0.0006167
6: 0.0045899, 0.0056422, 0.0046183, 0.0056941, -0.0006136, 0.0005598
7: -0.0062527, -0.0023258, -0.0061467, -0.0021323, -0.0022898, 0.0020890
8: -0.0073827, -0.0043263, -0.0075333, -0.0044089, -0.0016259, 0.0017822
9: -0.0036365, -0.0033728, -0.0036294, -0.0033598, -0.0001538, 0.0001403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003932
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003987
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088668, -0.0061883, -0.0090080, -0.0062707, -0.0014286, 0.0015997
1: -0.0054385, -0.0046834, -0.0054783, -0.0047066, -0.0004028, 0.0004510
2: -0.0015669, 0.0040049, -0.0018606, 0.0038334, -0.0029717, 0.0033278
3: 0.0014199, 0.0021573, 0.0013811, 0.0021346, -0.0003933, 0.0004404
4: 0.0030989, 0.0072629, 0.0032270, 0.0074824, -0.0024870, 0.0022209
5: 0.9963672, 0.9975241, 0.9964028, 0.9975851, -0.0006910, 0.0006170
6: 0.0045862, 0.0056363, 0.0046185, 0.0056916, -0.0006272, 0.0005601
7: -0.0062668, -0.0023479, -0.0061462, -0.0021414, -0.0023405, 0.0020901
8: -0.0073655, -0.0043154, -0.0075262, -0.0044093, -0.0016267, 0.0018216
9: -0.0036374, -0.0033743, -0.0036293, -0.0033604, -0.0001572, 0.0001403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003937
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003994
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090149, -0.0062511, -0.0090135, -0.0063022, -0.0015763, 0.0015856
1: -0.0054803, -0.0047011, -0.0054799, -0.0047155, -0.0004444, 0.0004470
2: -0.0018749, 0.0038742, -0.0018721, 0.0037680, -0.0032790, 0.0032984
3: 0.0013792, 0.0021400, 0.0013795, 0.0021259, -0.0004339, 0.0004365
4: 0.0031965, 0.0074930, 0.0032759, 0.0074910, -0.0024650, 0.0024505
5: 0.9963944, 0.9975880, 0.9964164, 0.9975874, -0.0006848, 0.0006808
6: 0.0046108, 0.0056943, 0.0046308, 0.0056938, -0.0006216, 0.0006180
7: -0.0061749, -0.0021314, -0.0061001, -0.0021333, -0.0023198, 0.0023062
8: -0.0075340, -0.0043870, -0.0075325, -0.0044451, -0.0017949, 0.0018055
9: -0.0036313, -0.0033597, -0.0036262, -0.0033599, -0.0001558, 0.0001549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003735, upper bound: 0.0003705
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003897, upper bound: 0.0003810
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0089949, -0.0062418, -0.0090073, -0.0063025, -0.0015737, 0.0016211
1: -0.0054746, -0.0046984, -0.0054781, -0.0047156, -0.0004437, 0.0004570
2: -0.0018333, 0.0038937, -0.0018591, 0.0037673, -0.0032736, 0.0033722
3: 0.0013847, 0.0021426, 0.0013813, 0.0021258, -0.0004332, 0.0004463
4: 0.0031819, 0.0074620, 0.0032764, 0.0074813, -0.0025202, 0.0024465
5: 0.9963902, 0.9975794, 0.9964165, 0.9975848, -0.0007002, 0.0006797
6: 0.0046071, 0.0056865, 0.0046309, 0.0056913, -0.0006355, 0.0006170
7: -0.0061886, -0.0021606, -0.0060997, -0.0021424, -0.0023717, 0.0023024
8: -0.0075113, -0.0043763, -0.0075254, -0.0044455, -0.0017920, 0.0018459
9: -0.0036322, -0.0033617, -0.0036262, -0.0033605, -0.0001593, 0.0001546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003735, upper bound: 0.0003723
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003897, upper bound: 0.0003826
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0088824, -0.0061981, -0.0090324, -0.0062365, -0.0014576, 0.0015771
1: -0.0054429, -0.0046861, -0.0054852, -0.0046970, -0.0004110, 0.0004446
2: -0.0015994, 0.0039846, -0.0019113, 0.0039047, -0.0030322, 0.0032807
3: 0.0014156, 0.0021546, 0.0013744, 0.0021440, -0.0004013, 0.0004342
4: 0.0031140, 0.0072871, 0.0031738, 0.0075203, -0.0024518, 0.0022660
5: 0.9963714, 0.9975308, 0.9963880, 0.9975956, -0.0006812, 0.0006296
6: 0.0045900, 0.0056424, 0.0046050, 0.0057012, -0.0006183, 0.0005715
7: -0.0062525, -0.0023251, -0.0061963, -0.0021057, -0.0023074, 0.0021326
8: -0.0073832, -0.0043265, -0.0075540, -0.0043703, -0.0016598, 0.0017959
9: -0.0036365, -0.0033727, -0.0036327, -0.0033580, -0.0001549, 0.0001432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003886, upper bound: 0.0003924
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003886, upper bound: 0.0003983
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088674, -0.0061885, -0.0090250, -0.0062368, -0.0014583, 0.0016110
1: -0.0054387, -0.0046834, -0.0054831, -0.0046971, -0.0004112, 0.0004542
2: -0.0015681, 0.0040045, -0.0018960, 0.0039040, -0.0030336, 0.0033511
3: 0.0014198, 0.0021572, 0.0013764, 0.0021439, -0.0004014, 0.0004435
4: 0.0030992, 0.0072638, 0.0031743, 0.0075088, -0.0025044, 0.0022671
5: 0.9963673, 0.9975243, 0.9963882, 0.9975923, -0.0006958, 0.0006299
6: 0.0045862, 0.0056365, 0.0046052, 0.0056983, -0.0006316, 0.0005717
7: -0.0062665, -0.0023471, -0.0061958, -0.0021165, -0.0023569, 0.0021336
8: -0.0073661, -0.0043157, -0.0075456, -0.0043707, -0.0016606, 0.0018344
9: -0.0036374, -0.0033742, -0.0036327, -0.0033587, -0.0001583, 0.0001433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003885, upper bound: 0.0003930
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003885, upper bound: 0.0003990
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090134, -0.0062513, -0.0090316, -0.0062716, -0.0016015, 0.0015954
1: -0.0054799, -0.0047011, -0.0054850, -0.0047069, -0.0004515, 0.0004498
2: -0.0018719, 0.0038739, -0.0019097, 0.0038316, -0.0033314, 0.0033187
3: 0.0013796, 0.0021399, 0.0013746, 0.0021344, -0.0004409, 0.0004392
4: 0.0031968, 0.0074908, 0.0032283, 0.0075191, -0.0024802, 0.0024897
5: 0.9963944, 0.9975874, 0.9964032, 0.9975953, -0.0006891, 0.0006917
6: 0.0046109, 0.0056937, 0.0046188, 0.0057009, -0.0006255, 0.0006279
7: -0.0061746, -0.0021334, -0.0061449, -0.0021069, -0.0023341, 0.0023431
8: -0.0075324, -0.0043872, -0.0075531, -0.0044103, -0.0018236, 0.0018167
9: -0.0036312, -0.0033599, -0.0036292, -0.0033581, -0.0001567, 0.0001573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003754, upper bound: 0.0003664
time: 1.01 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003922, upper bound: 0.0003797
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0089933, -0.0062420, -0.0090243, -0.0062719, -0.0015989, 0.0016300
1: -0.0054742, -0.0046985, -0.0054829, -0.0047069, -0.0004508, 0.0004595
2: -0.0018300, 0.0038933, -0.0018945, 0.0038310, -0.0033260, 0.0033907
3: 0.0013851, 0.0021425, 0.0013766, 0.0021343, -0.0004401, 0.0004487
4: 0.0031822, 0.0074595, 0.0032288, 0.0075077, -0.0025340, 0.0024856
5: 0.9963904, 0.9975787, 0.9964033, 0.9975921, -0.0007040, 0.0006906
6: 0.0046072, 0.0056858, 0.0046189, 0.0056980, -0.0006390, 0.0006268
7: -0.0061883, -0.0021629, -0.0061444, -0.0021176, -0.0023847, 0.0023393
8: -0.0075094, -0.0043765, -0.0075447, -0.0044106, -0.0018206, 0.0018560
9: -0.0036322, -0.0033619, -0.0036292, -0.0033588, -0.0001601, 0.0001571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003754, upper bound: 0.0003681
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003922, upper bound: 0.0003814
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090264, -0.0062699, -0.0089576, -0.0062477, -0.0017110, 0.0016163
1: -0.0054835, -0.0047064, -0.0054641, -0.0047001, -0.0004824, 0.0004557
2: -0.0018990, 0.0038352, -0.0017558, 0.0038815, -0.0035592, 0.0033623
3: 0.0013760, 0.0021348, 0.0013949, 0.0021409, -0.0004710, 0.0004449
4: 0.0032257, 0.0075111, 0.0031911, 0.0074040, -0.0025127, 0.0026600
5: 0.9964024, 0.9975930, 0.9963928, 0.9975633, -0.0006981, 0.0007390
6: 0.0046181, 0.0056988, 0.0046094, 0.0056719, -0.0006337, 0.0006708
7: -0.0061474, -0.0021144, -0.0061800, -0.0022151, -0.0023648, 0.0025033
8: -0.0075472, -0.0044083, -0.0074688, -0.0043830, -0.0019483, 0.0018405
9: -0.0036294, -0.0033586, -0.0036316, -0.0033654, -0.0001588, 0.0001681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003967, upper bound: 0.0003924
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003971, upper bound: 0.0003923
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090257, -0.0063017, -0.0090893, -0.0063001, -0.0017290, 0.0017568
1: -0.0054833, -0.0047153, -0.0055013, -0.0047149, -0.0004875, 0.0004953
2: -0.0018974, 0.0037690, -0.0020298, 0.0037724, -0.0035967, 0.0036546
3: 0.0013762, 0.0021261, 0.0013587, 0.0021265, -0.0004760, 0.0004836
4: 0.0032751, 0.0075099, 0.0032726, 0.0076088, -0.0027312, 0.0026880
5: 0.9964162, 0.9975927, 0.9964154, 0.9976202, -0.0007588, 0.0007468
6: 0.0046306, 0.0056986, 0.0046300, 0.0057235, -0.0006888, 0.0006779
7: -0.0061009, -0.0021155, -0.0061032, -0.0020224, -0.0025704, 0.0025297
8: -0.0075464, -0.0044446, -0.0076188, -0.0044427, -0.0019688, 0.0020005
9: -0.0036263, -0.0033587, -0.0036264, -0.0033524, -0.0001726, 0.0001699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003841, upper bound: 0.0003895
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003861, upper bound: 0.0003895
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090446, -0.0062360, -0.0089569, -0.0062478, -0.0017234, 0.0016575
1: -0.0054887, -0.0046968, -0.0054640, -0.0047002, -0.0004859, 0.0004673
2: -0.0019368, 0.0039057, -0.0017544, 0.0038811, -0.0035850, 0.0034480
3: 0.0013710, 0.0021442, 0.0013951, 0.0021409, -0.0004744, 0.0004563
4: 0.0031730, 0.0075393, 0.0031914, 0.0074030, -0.0025768, 0.0026792
5: 0.9963878, 0.9976009, 0.9963929, 0.9975630, -0.0007159, 0.0007444
6: 0.0046048, 0.0057060, 0.0046095, 0.0056716, -0.0006498, 0.0006757
7: -0.0061970, -0.0020878, -0.0061797, -0.0022161, -0.0024251, 0.0025214
8: -0.0075679, -0.0043697, -0.0074681, -0.0043832, -0.0019624, 0.0018874
9: -0.0036327, -0.0033568, -0.0036316, -0.0033654, -0.0001628, 0.0001693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003962, upper bound: 0.0003940
time: 0.87 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003964, upper bound: 0.0003941
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090438, -0.0062711, -0.0090889, -0.0063002, -0.0017394, 0.0017944
1: -0.0054884, -0.0047067, -0.0055012, -0.0047149, -0.0004904, 0.0005059
2: -0.0019351, 0.0038327, -0.0020289, 0.0037721, -0.0036182, 0.0037327
3: 0.0013712, 0.0021345, 0.0013588, 0.0021265, -0.0004788, 0.0004940
4: 0.0032276, 0.0075381, 0.0032729, 0.0076082, -0.0027896, 0.0027040
5: 0.9964029, 0.9976006, 0.9964155, 0.9976200, -0.0007750, 0.0007513
6: 0.0046186, 0.0057057, 0.0046300, 0.0057233, -0.0007035, 0.0006819
7: -0.0061456, -0.0020890, -0.0061030, -0.0020230, -0.0026253, 0.0025448
8: -0.0075670, -0.0044097, -0.0076184, -0.0044429, -0.0019806, 0.0020433
9: -0.0036293, -0.0033569, -0.0036264, -0.0033525, -0.0001763, 0.0001709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003827, upper bound: 0.0003903
time: 0.89 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003847, upper bound: 0.0003903
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0089325, -0.0061563, -0.0089212, -0.0062483, -0.0015346, 0.0015957
1: -0.0054571, -0.0046743, -0.0054539, -0.0047003, -0.0004327, 0.0004499
2: -0.0017036, 0.0040715, -0.0016801, 0.0038800, -0.0031922, 0.0033194
3: 0.0014019, 0.0021661, 0.0014050, 0.0021408, -0.0004224, 0.0004393
4: 0.0030491, 0.0073650, 0.0031922, 0.0073475, -0.0024807, 0.0023857
5: 0.9963534, 0.9975525, 0.9963931, 0.9975476, -0.0006892, 0.0006628
6: 0.0045736, 0.0056620, 0.0046097, 0.0056576, -0.0006256, 0.0006016
7: -0.0063136, -0.0022518, -0.0061789, -0.0022683, -0.0023346, 0.0022452
8: -0.0074403, -0.0042790, -0.0074274, -0.0043838, -0.0017474, 0.0018170
9: -0.0036406, -0.0033678, -0.0036315, -0.0033689, -0.0001568, 0.0001508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004423
time: 1.01 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004437
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0089316, -0.0061853, -0.0090529, -0.0063009, -0.0015556, 0.0017234
1: -0.0054568, -0.0046825, -0.0054910, -0.0047151, -0.0004386, 0.0004859
2: -0.0017017, 0.0040111, -0.0019541, 0.0037708, -0.0032360, 0.0035850
3: 0.0014021, 0.0021581, 0.0013687, 0.0021263, -0.0004282, 0.0004744
4: 0.0030942, 0.0073636, 0.0032738, 0.0075522, -0.0026792, 0.0024184
5: 0.9963659, 0.9975521, 0.9964157, 0.9976045, -0.0007444, 0.0006719
6: 0.0045850, 0.0056617, 0.0046303, 0.0057092, -0.0006757, 0.0006099
7: -0.0062711, -0.0022532, -0.0061021, -0.0020756, -0.0025214, 0.0022760
8: -0.0074392, -0.0043120, -0.0075774, -0.0044436, -0.0017714, 0.0019624
9: -0.0036377, -0.0033679, -0.0036264, -0.0033560, -0.0001693, 0.0001528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004432, upper bound: 0.0004474
time: 0.97 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004474
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0089320, -0.0061565, -0.0089296, -0.0062241, -0.0015717, 0.0016042
1: -0.0054569, -0.0046744, -0.0054562, -0.0046935, -0.0004431, 0.0004523
2: -0.0017026, 0.0040711, -0.0016974, 0.0039305, -0.0032695, 0.0033370
3: 0.0014020, 0.0021660, 0.0014027, 0.0021474, -0.0004327, 0.0004416
4: 0.0030493, 0.0073643, 0.0031545, 0.0073604, -0.0024939, 0.0024435
5: 0.9963534, 0.9975522, 0.9963827, 0.9975511, -0.0006929, 0.0006789
6: 0.0045737, 0.0056618, 0.0046002, 0.0056609, -0.0006289, 0.0006162
7: -0.0063134, -0.0022525, -0.0062144, -0.0022562, -0.0023470, 0.0022996
8: -0.0074397, -0.0042792, -0.0074369, -0.0043562, -0.0017898, 0.0018267
9: -0.0036405, -0.0033679, -0.0036339, -0.0033681, -0.0001576, 0.0001544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004493, upper bound: 0.0004423
time: 0.91 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004493, upper bound: 0.0004436
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0089311, -0.0061855, -0.0090611, -0.0062784, -0.0015937, 0.0017313
1: -0.0054567, -0.0046826, -0.0054933, -0.0047088, -0.0004493, 0.0004881
2: -0.0017006, 0.0040107, -0.0019710, 0.0038176, -0.0033152, 0.0036014
3: 0.0014022, 0.0021580, 0.0013665, 0.0021325, -0.0004387, 0.0004766
4: 0.0030945, 0.0073628, 0.0032388, 0.0075649, -0.0026915, 0.0024776
5: 0.9963660, 0.9975518, 0.9964061, 0.9976080, -0.0007478, 0.0006883
6: 0.0045851, 0.0056615, 0.0046215, 0.0057124, -0.0006788, 0.0006248
7: -0.0062709, -0.0022539, -0.0061350, -0.0020638, -0.0025330, 0.0023317
8: -0.0074386, -0.0043122, -0.0075866, -0.0044180, -0.0018147, 0.0019714
9: -0.0036377, -0.0033680, -0.0036286, -0.0033552, -0.0001701, 0.0001566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004498, upper bound: 0.0004474
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004493, upper bound: 0.0004474
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090264, -0.0062699, -0.0089312, -0.0061949, -0.0016470, 0.0014891
1: -0.0054835, -0.0047064, -0.0054567, -0.0046852, -0.0004643, 0.0004198
2: -0.0018990, 0.0038352, -0.0017009, 0.0039912, -0.0034261, 0.0030976
3: 0.0013760, 0.0021348, 0.0014022, 0.0021555, -0.0004534, 0.0004099
4: 0.0032257, 0.0075111, 0.0031091, 0.0073630, -0.0023149, 0.0025604
5: 0.9964024, 0.9975930, 0.9963701, 0.9975520, -0.0006432, 0.0007114
6: 0.0046181, 0.0056988, 0.0045887, 0.0056615, -0.0005838, 0.0006457
7: -0.0061474, -0.0021144, -0.0062571, -0.0022537, -0.0021786, 0.0024097
8: -0.0075472, -0.0044083, -0.0074388, -0.0043229, -0.0018754, 0.0016956
9: -0.0036294, -0.0033586, -0.0036368, -0.0033680, -0.0001463, 0.0001618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004000, upper bound: 0.0003979
time: 0.83 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004003, upper bound: 0.0003979
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090257, -0.0063017, -0.0090657, -0.0062452, -0.0016672, 0.0016357
1: -0.0054833, -0.0047153, -0.0054946, -0.0046994, -0.0004700, 0.0004612
2: -0.0018974, 0.0037690, -0.0019807, 0.0038865, -0.0034680, 0.0034026
3: 0.0013762, 0.0021261, 0.0013652, 0.0021416, -0.0004589, 0.0004503
4: 0.0032751, 0.0075099, 0.0031873, 0.0075721, -0.0025429, 0.0025918
5: 0.9964162, 0.9975927, 0.9963918, 0.9976100, -0.0007065, 0.0007201
6: 0.0046306, 0.0056986, 0.0046085, 0.0057143, -0.0006413, 0.0006536
7: -0.0061009, -0.0021155, -0.0061835, -0.0020569, -0.0023931, 0.0024392
8: -0.0075464, -0.0044446, -0.0075920, -0.0043802, -0.0018984, 0.0018626
9: -0.0036263, -0.0033587, -0.0036318, -0.0033547, -0.0001607, 0.0001638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003892, upper bound: 0.0003961
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003910, upper bound: 0.0003961
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090446, -0.0062360, -0.0089307, -0.0061951, -0.0016592, 0.0015192
1: -0.0054887, -0.0046968, -0.0054566, -0.0046853, -0.0004678, 0.0004283
2: -0.0019368, 0.0039057, -0.0016998, 0.0039908, -0.0034516, 0.0031603
3: 0.0013710, 0.0021442, 0.0014023, 0.0021554, -0.0004568, 0.0004182
4: 0.0031730, 0.0075393, 0.0031094, 0.0073622, -0.0023618, 0.0025795
5: 0.9963878, 0.9976009, 0.9963702, 0.9975517, -0.0006562, 0.0007167
6: 0.0046048, 0.0057060, 0.0045888, 0.0056613, -0.0005956, 0.0006505
7: -0.0061970, -0.0020878, -0.0062568, -0.0022545, -0.0022227, 0.0024276
8: -0.0075679, -0.0043697, -0.0074382, -0.0043232, -0.0018894, 0.0017299
9: -0.0036327, -0.0033568, -0.0036368, -0.0033680, -0.0001492, 0.0001630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003995, upper bound: 0.0004001
time: 0.84 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003999, upper bound: 0.0004001
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090438, -0.0062711, -0.0090639, -0.0062454, -0.0016772, 0.0016607
1: -0.0054884, -0.0047067, -0.0054941, -0.0046995, -0.0004729, 0.0004682
2: -0.0019351, 0.0038327, -0.0019769, 0.0038861, -0.0034888, 0.0034545
3: 0.0013712, 0.0021345, 0.0013657, 0.0021416, -0.0004617, 0.0004571
4: 0.0032276, 0.0075381, 0.0031876, 0.0075693, -0.0025817, 0.0026073
5: 0.9964029, 0.9976006, 0.9963918, 0.9976092, -0.0007173, 0.0007244
6: 0.0046186, 0.0057057, 0.0046085, 0.0057135, -0.0006511, 0.0006575
7: -0.0061456, -0.0020890, -0.0061832, -0.0020596, -0.0024296, 0.0024538
8: -0.0075670, -0.0044097, -0.0075899, -0.0043804, -0.0019098, 0.0018910
9: -0.0036293, -0.0033569, -0.0036318, -0.0033549, -0.0001631, 0.0001648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003880, upper bound: 0.0003980
time: 0.90 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003901, upper bound: 0.0003980
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0088956, -0.0061956, -0.0089325, -0.0061563, -0.0014724, 0.0014712
1: -0.0054467, -0.0046854, -0.0054571, -0.0046743, -0.0004151, 0.0004148
2: -0.0016268, 0.0039897, -0.0017036, 0.0040715, -0.0030628, 0.0030605
3: 0.0014120, 0.0021553, 0.0014019, 0.0021661, -0.0004053, 0.0004050
4: 0.0031102, 0.0073077, 0.0030491, 0.0073650, -0.0022872, 0.0022890
5: 0.9963703, 0.9975365, 0.9963534, 0.9975525, -0.0006355, 0.0006359
6: 0.0045890, 0.0056476, 0.0045736, 0.0056620, -0.0005768, 0.0005772
7: -0.0062561, -0.0023058, -0.0063136, -0.0022518, -0.0021525, 0.0021542
8: -0.0073982, -0.0043237, -0.0074403, -0.0042790, -0.0016766, 0.0016753
9: -0.0036367, -0.0033715, -0.0036406, -0.0033678, -0.0001445, 0.0001446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B2_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004391
time: 0.93 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004398
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0090310, -0.0062460, -0.0089316, -0.0061853, -0.0016024, 0.0014965
1: -0.0054848, -0.0046996, -0.0054568, -0.0046825, -0.0004518, 0.0004219
2: -0.0019084, 0.0038850, -0.0017017, 0.0040111, -0.0033333, 0.0031130
3: 0.0013748, 0.0021414, 0.0014021, 0.0021581, -0.0004411, 0.0004120
4: 0.0031885, 0.0075181, 0.0030942, 0.0073636, -0.0023265, 0.0024911
5: 0.9963921, 0.9975950, 0.9963659, 0.9975521, -0.0006464, 0.0006921
6: 0.0046088, 0.0057006, 0.0045850, 0.0056617, -0.0005867, 0.0006282
7: -0.0061824, -0.0021078, -0.0062711, -0.0022532, -0.0021895, 0.0023444
8: -0.0075524, -0.0043811, -0.0074392, -0.0043120, -0.0018247, 0.0017041
9: -0.0036318, -0.0033582, -0.0036377, -0.0033679, -0.0001470, 0.0001574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B2_A2_A1_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004497, upper bound: 0.0004391
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004496, upper bound: 0.0004398
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0089050, -0.0061681, -0.0089320, -0.0061565, -0.0014822, 0.0015052
1: -0.0054493, -0.0046777, -0.0054569, -0.0046744, -0.0004179, 0.0004244
2: -0.0016464, 0.0040470, -0.0017026, 0.0040711, -0.0030834, 0.0031312
3: 0.0014094, 0.0021628, 0.0014020, 0.0021660, -0.0004080, 0.0004144
4: 0.0030674, 0.0073222, 0.0030493, 0.0073643, -0.0023400, 0.0023043
5: 0.9963585, 0.9975406, 0.9963534, 0.9975522, -0.0006501, 0.0006402
6: 0.0045782, 0.0056512, 0.0045737, 0.0056618, -0.0005901, 0.0005811
7: -0.0062964, -0.0022921, -0.0063134, -0.0022525, -0.0022022, 0.0021686
8: -0.0074089, -0.0042924, -0.0074397, -0.0042792, -0.0016878, 0.0017140
9: -0.0036394, -0.0033705, -0.0036405, -0.0033679, -0.0001479, 0.0001456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004467
time: 1.06 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004474
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0090352, -0.0062207, -0.0089311, -0.0061855, -0.0016119, 0.0015312
1: -0.0054860, -0.0046925, -0.0054567, -0.0046826, -0.0004545, 0.0004317
2: -0.0019171, 0.0039376, -0.0017006, 0.0040107, -0.0033531, 0.0031852
3: 0.0013736, 0.0021484, 0.0014022, 0.0021580, -0.0004437, 0.0004215
4: 0.0031492, 0.0075246, 0.0030945, 0.0073628, -0.0023804, 0.0025059
5: 0.9963812, 0.9975968, 0.9963660, 0.9975518, -0.0006614, 0.0006962
6: 0.0045988, 0.0057023, 0.0045851, 0.0056615, -0.0006003, 0.0006320
7: -0.0062194, -0.0021017, -0.0062709, -0.0022539, -0.0022403, 0.0023583
8: -0.0075571, -0.0043523, -0.0074386, -0.0043122, -0.0018355, 0.0017436
9: -0.0036342, -0.0033577, -0.0036377, -0.0033680, -0.0001504, 0.0001584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004497, upper bound: 0.0004467
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004497, upper bound: 0.0004474
time: 0.87 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.29 seconds
IS_A1_B1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003868
IS_A1_B1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003930
IS_A1_B1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003868
IS_A1_B1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003933
IS_A1_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003638, upper bound: 0.0003664
IS_A1_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003804, upper bound: 0.0003746
IS_A1_B1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003638, upper bound: 0.0003673
IS_A1_B1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003804, upper bound: 0.0003760
IS_A1_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003854
IS_A1_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003919
IS_A1_B1_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003852
IS_A1_B1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003920
IS_A1_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003641, upper bound: 0.0003600
IS_A1_B1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003815, upper bound: 0.0003722
IS_A1_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003641, upper bound: 0.0003615
IS_A1_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003816, upper bound: 0.0003732
IS_A1_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003922
IS_A1_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003968
IS_A1_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003925
IS_A1_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003972
IS_A1_B1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003638, upper bound: 0.0003677
IS_A1_B1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003804, upper bound: 0.0003758
IS_A1_B1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003639, upper bound: 0.0003688
IS_A1_B1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003804, upper bound: 0.0003777
IS_A1_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003910
IS_A1_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003962
IS_A1_B1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003913
IS_A1_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003796, upper bound: 0.0003964
IS_A1_B1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003641, upper bound: 0.0003625
IS_A1_B1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003816, upper bound: 0.0003743
IS_A1_B1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003641, upper bound: 0.0003635
IS_A1_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003816, upper bound: 0.0003762
IS_A1_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003929, upper bound: 0.0003923
IS_A1_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003934, upper bound: 0.0003923
IS_A1_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003825, upper bound: 0.0003895
IS_A1_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003841, upper bound: 0.0003895
IS_A1_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003920, upper bound: 0.0003941
IS_A1_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003920, upper bound: 0.0003941
IS_A1_B2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003903
IS_A1_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003815, upper bound: 0.0003903
IS_A1_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003929, upper bound: 0.0003972
IS_A1_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003934, upper bound: 0.0003972
IS_A1_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003824, upper bound: 0.0003958
IS_A1_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003958
IS_A1_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003919, upper bound: 0.0004000
IS_A1_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003921, upper bound: 0.0004000
IS_A1_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003806, upper bound: 0.0003983
IS_A1_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003820, upper bound: 0.0003983
IS_A1_B2_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004387
IS_A1_B2_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004406
IS_A1_B2_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004387
IS_A1_B2_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004405
IS_A1_B2_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004461
IS_A1_B2_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004474
IS_A1_B2_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004461
IS_A1_B2_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004474
IS_A1_B2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004423, upper bound: 0.0004401
IS_A1_B2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004401
IS_A1_B2_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004401
IS_A1_B2_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004401
IS_A1_B2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004423, upper bound: 0.0004486
IS_A1_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004486
IS_A1_B2_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004486
IS_A1_B2_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004486
IS_A2_B1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003868
IS_A2_B1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003930
IS_A2_B1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003868
IS_A2_B1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003934
IS_A2_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003702, upper bound: 0.0003664
IS_A2_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003866, upper bound: 0.0003745
IS_A2_B1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003702, upper bound: 0.0003675
IS_A2_B1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003866, upper bound: 0.0003763
IS_A2_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003854
IS_A2_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003920
IS_A2_B1_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003843, upper bound: 0.0003852
IS_A2_B1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003843, upper bound: 0.0003921
IS_A2_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003714, upper bound: 0.0003602
IS_A2_B1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003725
IS_A2_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003714, upper bound: 0.0003617
IS_A2_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003738
IS_A2_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003932
IS_A2_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003987
IS_A2_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003937
IS_A2_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003994
IS_A2_B1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003735, upper bound: 0.0003705
IS_A2_B1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003897, upper bound: 0.0003810
IS_A2_B1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003735, upper bound: 0.0003723
IS_A2_B1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003897, upper bound: 0.0003826
IS_A2_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003886, upper bound: 0.0003924
IS_A2_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003886, upper bound: 0.0003983
IS_A2_B1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003885, upper bound: 0.0003930
IS_A2_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003885, upper bound: 0.0003990
IS_A2_B1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003754, upper bound: 0.0003664
IS_A2_B1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003922, upper bound: 0.0003797
IS_A2_B1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003754, upper bound: 0.0003681
IS_A2_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003922, upper bound: 0.0003814
IS_A2_B2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003967, upper bound: 0.0003924
IS_A2_B2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003971, upper bound: 0.0003923
IS_A2_B2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003841, upper bound: 0.0003895
IS_A2_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003861, upper bound: 0.0003895
IS_A2_B2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003962, upper bound: 0.0003940
IS_A2_B2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003964, upper bound: 0.0003941
IS_A2_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003827, upper bound: 0.0003903
IS_A2_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003847, upper bound: 0.0003903
IS_A2_B2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004423
IS_A2_B2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004437
IS_A2_B2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004432, upper bound: 0.0004474
IS_A2_B2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004474
IS_A2_B2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004493, upper bound: 0.0004423
IS_A2_B2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004493, upper bound: 0.0004436
IS_A2_B2_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004498, upper bound: 0.0004474
IS_A2_B2_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004493, upper bound: 0.0004474
IS_A2_B2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004000, upper bound: 0.0003979
IS_A2_B2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004003, upper bound: 0.0003979
IS_A2_B2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003892, upper bound: 0.0003961
IS_A2_B2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003910, upper bound: 0.0003961
IS_A2_B2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003995, upper bound: 0.0004001
IS_A2_B2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003999, upper bound: 0.0004001
IS_A2_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003880, upper bound: 0.0003980
IS_A2_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0003901, upper bound: 0.0003980
IS_A2_B2_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004391
IS_A2_B2_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004398
IS_A2_B2_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004497, upper bound: 0.0004391
IS_A2_B2_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004496, upper bound: 0.0004398
IS_A2_B2_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004467
IS_A2_B2_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004474
IS_A2_B2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004497, upper bound: 0.0004467
IS_A2_B2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0004497, upper bound: 0.0004474

## BFS IS instance: IS_A1_B1_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0090476, -0.0063569, -0.0090431, -0.0063213, -0.0014329, 0.0014085
1: -0.0054895, -0.0047309, -0.0054883, -0.0047209, -0.0004040, 0.0003971
2: -0.0019429, 0.0036541, -0.0019337, 0.0037283, -0.0029807, 0.0029300
3: 0.0013702, 0.0021109, 0.0013714, 0.0021207, -0.0003945, 0.0003877
4: 0.0033610, 0.0075439, 0.0033055, 0.0075370, -0.0021897, 0.0022276
5: 0.9964400, 0.9976021, 0.9964246, 0.9976003, -0.0006084, 0.0006189
6: 0.0046523, 0.0057071, 0.0046383, 0.0057054, -0.0005522, 0.0005618
7: -0.0060201, -0.0020835, -0.0060723, -0.0020900, -0.0020608, 0.0020964
8: -0.0075713, -0.0045074, -0.0075662, -0.0044668, -0.0016317, 0.0016039
9: -0.0036209, -0.0033565, -0.0036244, -0.0033570, -0.0001384, 0.0001408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003477, upper bound: 0.0003672
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003782
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0089074, -0.0062500, -0.0090431, -0.0063213, -0.0014181, 0.0016213
1: -0.0054500, -0.0047008, -0.0054883, -0.0047209, -0.0003998, 0.0004571
2: -0.0016513, 0.0038766, -0.0019337, 0.0037283, -0.0029499, 0.0033727
3: 0.0014088, 0.0021403, 0.0013714, 0.0021207, -0.0003904, 0.0004463
4: 0.0031947, 0.0073259, 0.0033055, 0.0075370, -0.0025206, 0.0022046
5: 0.9963938, 0.9975416, 0.9964246, 0.9976003, -0.0007003, 0.0006125
6: 0.0046103, 0.0056522, 0.0046383, 0.0057054, -0.0006356, 0.0005560
7: -0.0061766, -0.0022886, -0.0060723, -0.0020900, -0.0023721, 0.0020748
8: -0.0074116, -0.0043856, -0.0075662, -0.0044668, -0.0016148, 0.0018462
9: -0.0036314, -0.0033703, -0.0036244, -0.0033570, -0.0001593, 0.0001393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003477, upper bound: 0.0003766
time: 1.04 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003844
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090398, -0.0063422, -0.0090376, -0.0063216, -0.0014376, 0.0014445
1: -0.0054873, -0.0047268, -0.0054867, -0.0047209, -0.0004053, 0.0004072
2: -0.0019267, 0.0036848, -0.0019223, 0.0037277, -0.0029905, 0.0030048
3: 0.0013723, 0.0021149, 0.0013729, 0.0021206, -0.0003957, 0.0003976
4: 0.0033380, 0.0075318, 0.0033060, 0.0075285, -0.0022456, 0.0022349
5: 0.9964337, 0.9975988, 0.9964248, 0.9975979, -0.0006239, 0.0006209
6: 0.0046465, 0.0057041, 0.0046384, 0.0057032, -0.0005663, 0.0005636
7: -0.0060417, -0.0020949, -0.0060718, -0.0020980, -0.0021133, 0.0021033
8: -0.0075624, -0.0044906, -0.0075600, -0.0044672, -0.0016370, 0.0016448
9: -0.0036223, -0.0033573, -0.0036243, -0.0033575, -0.0001419, 0.0001412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003474, upper bound: 0.0003667
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003781
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0088969, -0.0062387, -0.0090376, -0.0063216, -0.0014228, 0.0016534
1: -0.0054470, -0.0046976, -0.0054867, -0.0047209, -0.0004011, 0.0004661
2: -0.0016296, 0.0039002, -0.0019223, 0.0037277, -0.0029597, 0.0034394
3: 0.0014116, 0.0021434, 0.0013729, 0.0021206, -0.0003917, 0.0004551
4: 0.0031771, 0.0073097, 0.0033060, 0.0075285, -0.0025704, 0.0022119
5: 0.9963889, 0.9975370, 0.9964248, 0.9975979, -0.0007141, 0.0006145
6: 0.0046059, 0.0056481, 0.0046384, 0.0057032, -0.0006482, 0.0005578
7: -0.0061931, -0.0023039, -0.0060718, -0.0020980, -0.0024190, 0.0020816
8: -0.0073997, -0.0043728, -0.0075600, -0.0044672, -0.0016201, 0.0018827
9: -0.0036325, -0.0033713, -0.0036243, -0.0033575, -0.0001624, 0.0001398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003474, upper bound: 0.0003764
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003848
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090177, -0.0063072, -0.0089847, -0.0063526, -0.0015270, 0.0015061
1: -0.0054811, -0.0047169, -0.0054718, -0.0047297, -0.0004305, 0.0004246
2: -0.0018808, 0.0037575, -0.0018121, 0.0036631, -0.0031765, 0.0031330
3: 0.0013784, 0.0021245, 0.0013875, 0.0021120, -0.0004204, 0.0004146
4: 0.0032837, 0.0074974, 0.0033543, 0.0074461, -0.0023414, 0.0023739
5: 0.9964185, 0.9975892, 0.9964381, 0.9975750, -0.0006505, 0.0006596
6: 0.0046328, 0.0056954, 0.0046506, 0.0056825, -0.0005905, 0.0005987
7: -0.0060928, -0.0021272, -0.0060264, -0.0021755, -0.0022035, 0.0022341
8: -0.0075372, -0.0044508, -0.0074997, -0.0045025, -0.0017388, 0.0017150
9: -0.0036257, -0.0033595, -0.0036213, -0.0033627, -0.0001480, 0.0001500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003216, upper bound: 0.0003387
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003216, upper bound: 0.0003664
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090283, -0.0063065, -0.0090114, -0.0063555, -0.0015584, 0.0015082
1: -0.0054841, -0.0047167, -0.0054793, -0.0047305, -0.0004394, 0.0004252
2: -0.0019029, 0.0037589, -0.0018677, 0.0036571, -0.0032418, 0.0031374
3: 0.0013755, 0.0021247, 0.0013801, 0.0021112, -0.0004290, 0.0004152
4: 0.0032827, 0.0075140, 0.0033588, 0.0074876, -0.0023447, 0.0024228
5: 0.9964183, 0.9975938, 0.9964395, 0.9975865, -0.0006514, 0.0006731
6: 0.0046325, 0.0056996, 0.0046517, 0.0056929, -0.0005913, 0.0006110
7: -0.0060938, -0.0021116, -0.0060221, -0.0021364, -0.0022066, 0.0022801
8: -0.0075494, -0.0044501, -0.0075301, -0.0045058, -0.0017746, 0.0017174
9: -0.0036258, -0.0033584, -0.0036210, -0.0033601, -0.0001482, 0.0001531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003482, upper bound: 0.0003506
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003482, upper bound: 0.0003746
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090028, -0.0062953, -0.0089795, -0.0063529, -0.0015274, 0.0015452
1: -0.0054769, -0.0047135, -0.0054703, -0.0047298, -0.0004306, 0.0004356
2: -0.0018498, 0.0037824, -0.0018014, 0.0036625, -0.0031774, 0.0032143
3: 0.0013825, 0.0021278, 0.0013889, 0.0021120, -0.0004205, 0.0004254
4: 0.0032652, 0.0074743, 0.0033548, 0.0074381, -0.0024022, 0.0023746
5: 0.9964134, 0.9975829, 0.9964383, 0.9975728, -0.0006674, 0.0006597
6: 0.0046281, 0.0056896, 0.0046507, 0.0056804, -0.0006058, 0.0005988
7: -0.0061103, -0.0021490, -0.0060259, -0.0021831, -0.0022607, 0.0022347
8: -0.0075203, -0.0044372, -0.0074938, -0.0045029, -0.0017393, 0.0017595
9: -0.0036269, -0.0033609, -0.0036212, -0.0033632, -0.0001518, 0.0001501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003215, upper bound: 0.0003402
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003215, upper bound: 0.0003674
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090132, -0.0062946, -0.0090057, -0.0063558, -0.0015587, 0.0015495
1: -0.0054798, -0.0047133, -0.0054777, -0.0047306, -0.0004395, 0.0004369
2: -0.0018715, 0.0037838, -0.0018559, 0.0036565, -0.0032425, 0.0032233
3: 0.0013796, 0.0021280, 0.0013817, 0.0021112, -0.0004291, 0.0004265
4: 0.0032641, 0.0074905, 0.0033592, 0.0074789, -0.0024089, 0.0024232
5: 0.9964131, 0.9975873, 0.9964395, 0.9975840, -0.0006693, 0.0006732
6: 0.0046278, 0.0056937, 0.0046518, 0.0056907, -0.0006075, 0.0006111
7: -0.0061113, -0.0021337, -0.0060217, -0.0021447, -0.0022670, 0.0022805
8: -0.0075322, -0.0044365, -0.0075236, -0.0045062, -0.0017750, 0.0017644
9: -0.0036270, -0.0033599, -0.0036210, -0.0033606, -0.0001522, 0.0001531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003482, upper bound: 0.0003527
time: 0.98 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003482, upper bound: 0.0003759
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0090507, -0.0063571, -0.0090610, -0.0062860, -0.0014688, 0.0014188
1: -0.0054904, -0.0047310, -0.0054933, -0.0047109, -0.0004141, 0.0004000
2: -0.0019494, 0.0036538, -0.0019709, 0.0038016, -0.0030553, 0.0029515
3: 0.0013693, 0.0021108, 0.0013665, 0.0021304, -0.0004043, 0.0003906
4: 0.0033612, 0.0075487, 0.0032508, 0.0075648, -0.0022058, 0.0022833
5: 0.9964401, 0.9976035, 0.9964094, 0.9976079, -0.0006128, 0.0006344
6: 0.0046523, 0.0057083, 0.0046245, 0.0057124, -0.0005563, 0.0005758
7: -0.0060199, -0.0020790, -0.0061238, -0.0020638, -0.0020759, 0.0021489
8: -0.0075748, -0.0045076, -0.0075866, -0.0044267, -0.0016725, 0.0016156
9: -0.0036208, -0.0033562, -0.0036278, -0.0033552, -0.0001394, 0.0001443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003490, upper bound: 0.0003638
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003766
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0089065, -0.0062501, -0.0090610, -0.0062860, -0.0014506, 0.0016316
1: -0.0054497, -0.0047008, -0.0054933, -0.0047109, -0.0004090, 0.0004600
2: -0.0016496, 0.0038763, -0.0019709, 0.0038016, -0.0030175, 0.0033941
3: 0.0014090, 0.0021403, 0.0013665, 0.0021304, -0.0003993, 0.0004492
4: 0.0031950, 0.0073247, 0.0032508, 0.0075648, -0.0025366, 0.0022551
5: 0.9963939, 0.9975412, 0.9964094, 0.9976079, -0.0007047, 0.0006265
6: 0.0046104, 0.0056518, 0.0046245, 0.0057124, -0.0006397, 0.0005687
7: -0.0061763, -0.0022898, -0.0061238, -0.0020638, -0.0023872, 0.0021223
8: -0.0074107, -0.0043858, -0.0075866, -0.0044267, -0.0016518, 0.0018579
9: -0.0036313, -0.0033704, -0.0036278, -0.0033552, -0.0001603, 0.0001425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003490, upper bound: 0.0003731
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003832
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090421, -0.0063423, -0.0090552, -0.0062863, -0.0014735, 0.0014550
1: -0.0054880, -0.0047268, -0.0054917, -0.0047110, -0.0004154, 0.0004102
2: -0.0019316, 0.0036845, -0.0019589, 0.0038010, -0.0030652, 0.0030267
3: 0.0013717, 0.0021149, 0.0013681, 0.0021303, -0.0004056, 0.0004005
4: 0.0033383, 0.0075354, 0.0032512, 0.0075558, -0.0022620, 0.0022907
5: 0.9964337, 0.9975998, 0.9964095, 0.9976054, -0.0006284, 0.0006364
6: 0.0046465, 0.0057050, 0.0046246, 0.0057101, -0.0005704, 0.0005777
7: -0.0060414, -0.0020914, -0.0061234, -0.0020723, -0.0021288, 0.0021558
8: -0.0075651, -0.0044908, -0.0075800, -0.0044270, -0.0016779, 0.0016568
9: -0.0036223, -0.0033571, -0.0036278, -0.0033558, -0.0001429, 0.0001448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003630
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003764
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0088967, -0.0062388, -0.0090552, -0.0062863, -0.0014552, 0.0016639
1: -0.0054470, -0.0046976, -0.0054917, -0.0047110, -0.0004103, 0.0004691
2: -0.0016292, 0.0038998, -0.0019589, 0.0038010, -0.0030272, 0.0034613
3: 0.0014117, 0.0021434, 0.0013681, 0.0021303, -0.0004006, 0.0004580
4: 0.0031774, 0.0073094, 0.0032512, 0.0075558, -0.0025867, 0.0022623
5: 0.9963890, 0.9975370, 0.9964095, 0.9976054, -0.0007187, 0.0006285
6: 0.0046060, 0.0056480, 0.0046246, 0.0057101, -0.0006523, 0.0005705
7: -0.0061929, -0.0023042, -0.0061234, -0.0020723, -0.0024344, 0.0021291
8: -0.0073995, -0.0043729, -0.0075800, -0.0044270, -0.0016571, 0.0018947
9: -0.0036325, -0.0033713, -0.0036278, -0.0033558, -0.0001635, 0.0001430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003726
time: 1.08 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003833
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090171, -0.0063074, -0.0090011, -0.0063214, -0.0015452, 0.0015137
1: -0.0054809, -0.0047170, -0.0054764, -0.0047209, -0.0004356, 0.0004268
2: -0.0018795, 0.0037572, -0.0018463, 0.0037281, -0.0032143, 0.0031487
3: 0.0013786, 0.0021245, 0.0013830, 0.0021207, -0.0004254, 0.0004167
4: 0.0032840, 0.0074965, 0.0033057, 0.0074717, -0.0023532, 0.0024022
5: 0.9964187, 0.9975891, 0.9964247, 0.9975821, -0.0006538, 0.0006674
6: 0.0046328, 0.0056952, 0.0046383, 0.0056889, -0.0005934, 0.0006058
7: -0.0060925, -0.0021281, -0.0060721, -0.0021515, -0.0022146, 0.0022607
8: -0.0075366, -0.0044510, -0.0075184, -0.0044669, -0.0017595, 0.0017236
9: -0.0036257, -0.0033595, -0.0036244, -0.0033611, -0.0001487, 0.0001518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003221, upper bound: 0.0003328
time: 0.98 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003221, upper bound: 0.0003601
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090278, -0.0063067, -0.0090321, -0.0063232, -0.0015855, 0.0015163
1: -0.0054839, -0.0047168, -0.0054851, -0.0047214, -0.0004470, 0.0004275
2: -0.0019018, 0.0037586, -0.0019107, 0.0037242, -0.0032981, 0.0031542
3: 0.0013756, 0.0021247, 0.0013744, 0.0021201, -0.0004365, 0.0004174
4: 0.0032829, 0.0075132, 0.0033086, 0.0075198, -0.0023572, 0.0024648
5: 0.9964183, 0.9975936, 0.9964254, 0.9975955, -0.0006549, 0.0006848
6: 0.0046326, 0.0056994, 0.0046390, 0.0057011, -0.0005945, 0.0006216
7: -0.0060936, -0.0021124, -0.0060694, -0.0021061, -0.0022184, 0.0023197
8: -0.0075488, -0.0044502, -0.0075537, -0.0044691, -0.0018054, 0.0017266
9: -0.0036258, -0.0033585, -0.0036242, -0.0033580, -0.0001490, 0.0001558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003499, upper bound: 0.0003482
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003499, upper bound: 0.0003721
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090019, -0.0062954, -0.0089961, -0.0063217, -0.0015456, 0.0015533
1: -0.0054766, -0.0047136, -0.0054750, -0.0047210, -0.0004358, 0.0004379
2: -0.0018480, 0.0037820, -0.0018359, 0.0037275, -0.0032152, 0.0032312
3: 0.0013827, 0.0021278, 0.0013843, 0.0021206, -0.0004255, 0.0004276
4: 0.0032654, 0.0074729, 0.0033062, 0.0074639, -0.0024148, 0.0024028
5: 0.9964135, 0.9975825, 0.9964247, 0.9975799, -0.0006709, 0.0006676
6: 0.0046282, 0.0056892, 0.0046384, 0.0056870, -0.0006090, 0.0006060
7: -0.0061100, -0.0021503, -0.0060717, -0.0021588, -0.0022726, 0.0022613
8: -0.0075193, -0.0044374, -0.0075127, -0.0044673, -0.0017600, 0.0017687
9: -0.0036269, -0.0033610, -0.0036243, -0.0033616, -0.0001526, 0.0001518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003221, upper bound: 0.0003341
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003221, upper bound: 0.0003613
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090125, -0.0062947, -0.0090266, -0.0063235, -0.0015860, 0.0015581
1: -0.0054796, -0.0047134, -0.0054836, -0.0047215, -0.0004471, 0.0004393
2: -0.0018700, 0.0037835, -0.0018994, 0.0037237, -0.0032991, 0.0032411
3: 0.0013798, 0.0021280, 0.0013759, 0.0021201, -0.0004366, 0.0004289
4: 0.0032643, 0.0074894, 0.0033090, 0.0075113, -0.0024222, 0.0024655
5: 0.9964131, 0.9975870, 0.9964256, 0.9975932, -0.0006730, 0.0006850
6: 0.0046279, 0.0056934, 0.0046392, 0.0056989, -0.0006108, 0.0006218
7: -0.0061111, -0.0021348, -0.0060690, -0.0021141, -0.0022795, 0.0023203
8: -0.0075314, -0.0044366, -0.0075474, -0.0044694, -0.0018059, 0.0017742
9: -0.0036270, -0.0033600, -0.0036241, -0.0033586, -0.0001531, 0.0001558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003499, upper bound: 0.0003500
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003499, upper bound: 0.0003732
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0090476, -0.0063569, -0.0090142, -0.0062704, -0.0015717, 0.0014797
1: -0.0054895, -0.0047309, -0.0054801, -0.0047065, -0.0004431, 0.0004172
2: -0.0019429, 0.0036541, -0.0018736, 0.0038341, -0.0032695, 0.0030780
3: 0.0013702, 0.0021109, 0.0013794, 0.0021347, -0.0004327, 0.0004073
4: 0.0033610, 0.0075439, 0.0032265, 0.0074921, -0.0023003, 0.0024434
5: 0.9964400, 0.9976021, 0.9964027, 0.9975877, -0.0006391, 0.0006789
6: 0.0046523, 0.0057071, 0.0046183, 0.0056941, -0.0005801, 0.0006162
7: -0.0060201, -0.0020835, -0.0061467, -0.0021323, -0.0021649, 0.0022995
8: -0.0075713, -0.0045074, -0.0075333, -0.0044089, -0.0017897, 0.0016849
9: -0.0036209, -0.0033565, -0.0036294, -0.0033598, -0.0001454, 0.0001544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003477, upper bound: 0.0003722
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003837
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0089074, -0.0062500, -0.0090142, -0.0062704, -0.0015569, 0.0016925
1: -0.0054500, -0.0047008, -0.0054801, -0.0047065, -0.0004390, 0.0004772
2: -0.0016513, 0.0038766, -0.0018736, 0.0038341, -0.0032387, 0.0035207
3: 0.0014088, 0.0021403, 0.0013794, 0.0021347, -0.0004286, 0.0004659
4: 0.0031947, 0.0073259, 0.0032265, 0.0074921, -0.0026312, 0.0024204
5: 0.9963938, 0.9975416, 0.9964027, 0.9975877, -0.0007310, 0.0006725
6: 0.0046103, 0.0056522, 0.0046183, 0.0056941, -0.0006635, 0.0006104
7: -0.0061766, -0.0022886, -0.0061467, -0.0021323, -0.0024762, 0.0022779
8: -0.0074116, -0.0043856, -0.0075333, -0.0044089, -0.0017729, 0.0019272
9: -0.0036314, -0.0033703, -0.0036294, -0.0033598, -0.0001663, 0.0001530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003477, upper bound: 0.0003791
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003881
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090398, -0.0063422, -0.0090080, -0.0062707, -0.0015766, 0.0015071
1: -0.0054873, -0.0047268, -0.0054783, -0.0047066, -0.0004445, 0.0004249
2: -0.0019267, 0.0036848, -0.0018606, 0.0038334, -0.0032797, 0.0031350
3: 0.0013723, 0.0021149, 0.0013811, 0.0021346, -0.0004340, 0.0004149
4: 0.0033380, 0.0075318, 0.0032270, 0.0074824, -0.0023429, 0.0024510
5: 0.9964337, 0.9975988, 0.9964028, 0.9975851, -0.0006509, 0.0006810
6: 0.0046465, 0.0057041, 0.0046185, 0.0056916, -0.0005908, 0.0006181
7: -0.0060417, -0.0020949, -0.0061462, -0.0021414, -0.0022049, 0.0023067
8: -0.0075624, -0.0044906, -0.0075262, -0.0044093, -0.0017953, 0.0017161
9: -0.0036223, -0.0033573, -0.0036293, -0.0033604, -0.0001481, 0.0001549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003474, upper bound: 0.0003723
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003840
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0088969, -0.0062387, -0.0090080, -0.0062707, -0.0015618, 0.0017160
1: -0.0054470, -0.0046976, -0.0054783, -0.0047066, -0.0004403, 0.0004838
2: -0.0016296, 0.0039002, -0.0018606, 0.0038334, -0.0032489, 0.0035696
3: 0.0014116, 0.0021434, 0.0013811, 0.0021346, -0.0004299, 0.0004724
4: 0.0031771, 0.0073097, 0.0032270, 0.0074824, -0.0026677, 0.0024280
5: 0.9963889, 0.9975370, 0.9964028, 0.9975851, -0.0007412, 0.0006746
6: 0.0046059, 0.0056481, 0.0046185, 0.0056916, -0.0006727, 0.0006123
7: -0.0061931, -0.0023039, -0.0061462, -0.0021414, -0.0025106, 0.0022850
8: -0.0073997, -0.0043728, -0.0075262, -0.0044093, -0.0017784, 0.0019540
9: -0.0036325, -0.0033713, -0.0036293, -0.0033604, -0.0001686, 0.0001534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003474, upper bound: 0.0003789
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003884
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090177, -0.0063072, -0.0089549, -0.0063042, -0.0016623, 0.0015901
1: -0.0054811, -0.0047169, -0.0054634, -0.0047160, -0.0004687, 0.0004483
2: -0.0018808, 0.0037575, -0.0017502, 0.0037638, -0.0034580, 0.0033077
3: 0.0013784, 0.0021245, 0.0013957, 0.0021254, -0.0004576, 0.0004377
4: 0.0032837, 0.0074974, 0.0032790, 0.0073998, -0.0024720, 0.0025843
5: 0.9964185, 0.9975892, 0.9964172, 0.9975621, -0.0006868, 0.0007180
6: 0.0046328, 0.0056954, 0.0046316, 0.0056708, -0.0006234, 0.0006517
7: -0.0060928, -0.0021272, -0.0060972, -0.0022191, -0.0023264, 0.0024321
8: -0.0075372, -0.0044508, -0.0074658, -0.0044474, -0.0018929, 0.0018107
9: -0.0036257, -0.0033595, -0.0036260, -0.0033656, -0.0001562, 0.0001633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003218, upper bound: 0.0003440
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003218, upper bound: 0.0003677
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090283, -0.0063065, -0.0089839, -0.0063047, -0.0016895, 0.0015836
1: -0.0054841, -0.0047167, -0.0054716, -0.0047162, -0.0004763, 0.0004465
2: -0.0019029, 0.0037589, -0.0018106, 0.0037628, -0.0035146, 0.0032942
3: 0.0013755, 0.0021247, 0.0013877, 0.0021252, -0.0004651, 0.0004359
4: 0.0032827, 0.0075140, 0.0032798, 0.0074450, -0.0024619, 0.0026266
5: 0.9964183, 0.9975938, 0.9964175, 0.9975746, -0.0006840, 0.0007297
6: 0.0046325, 0.0056996, 0.0046318, 0.0056822, -0.0006209, 0.0006624
7: -0.0060938, -0.0021116, -0.0060965, -0.0021766, -0.0023169, 0.0024719
8: -0.0075494, -0.0044501, -0.0074988, -0.0044480, -0.0019239, 0.0018033
9: -0.0036258, -0.0033584, -0.0036260, -0.0033628, -0.0001556, 0.0001660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003484, upper bound: 0.0003561
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003484, upper bound: 0.0003759
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090028, -0.0062953, -0.0089488, -0.0063045, -0.0016629, 0.0016181
1: -0.0054769, -0.0047135, -0.0054617, -0.0047161, -0.0004688, 0.0004562
2: -0.0018498, 0.0037824, -0.0017376, 0.0037632, -0.0034592, 0.0033660
3: 0.0013825, 0.0021278, 0.0013974, 0.0021253, -0.0004578, 0.0004454
4: 0.0032652, 0.0074743, 0.0032795, 0.0073904, -0.0025155, 0.0025852
5: 0.9964134, 0.9975829, 0.9964173, 0.9975595, -0.0006989, 0.0007182
6: 0.0046281, 0.0056896, 0.0046317, 0.0056684, -0.0006344, 0.0006519
7: -0.0061103, -0.0021490, -0.0060968, -0.0022279, -0.0023674, 0.0024329
8: -0.0075203, -0.0044372, -0.0074589, -0.0044477, -0.0018936, 0.0018425
9: -0.0036269, -0.0033609, -0.0036260, -0.0033662, -0.0001590, 0.0001634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003216, upper bound: 0.0003455
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003216, upper bound: 0.0003687
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090132, -0.0062946, -0.0089778, -0.0063050, -0.0016901, 0.0016153
1: -0.0054798, -0.0047133, -0.0054698, -0.0047163, -0.0004765, 0.0004554
2: -0.0018715, 0.0037838, -0.0017977, 0.0037622, -0.0035157, 0.0033601
3: 0.0013796, 0.0021280, 0.0013894, 0.0021252, -0.0004653, 0.0004447
4: 0.0032641, 0.0074905, 0.0032803, 0.0074354, -0.0025112, 0.0026274
5: 0.9964131, 0.9975873, 0.9964176, 0.9975720, -0.0006977, 0.0007300
6: 0.0046278, 0.0056937, 0.0046319, 0.0056798, -0.0006333, 0.0006626
7: -0.0061113, -0.0021337, -0.0060960, -0.0021856, -0.0023633, 0.0024727
8: -0.0075322, -0.0044365, -0.0074918, -0.0044483, -0.0019245, 0.0018393
9: -0.0036270, -0.0033599, -0.0036260, -0.0033634, -0.0001587, 0.0001660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003484, upper bound: 0.0003582
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003484, upper bound: 0.0003777
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0090507, -0.0063571, -0.0090324, -0.0062365, -0.0016155, 0.0014919
1: -0.0054904, -0.0047310, -0.0054852, -0.0046970, -0.0004555, 0.0004206
2: -0.0019494, 0.0036538, -0.0019113, 0.0039047, -0.0033605, 0.0031035
3: 0.0013693, 0.0021108, 0.0013744, 0.0021440, -0.0004447, 0.0004107
4: 0.0033612, 0.0075487, 0.0031738, 0.0075203, -0.0023194, 0.0025114
5: 0.9964401, 0.9976035, 0.9963880, 0.9975956, -0.0006444, 0.0006977
6: 0.0046523, 0.0057083, 0.0046050, 0.0057012, -0.0005849, 0.0006333
7: -0.0060199, -0.0020790, -0.0061963, -0.0021057, -0.0021828, 0.0023635
8: -0.0075748, -0.0045076, -0.0075540, -0.0043703, -0.0018395, 0.0016989
9: -0.0036208, -0.0033562, -0.0036327, -0.0033580, -0.0001466, 0.0001587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003490, upper bound: 0.0003693
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003824
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0089065, -0.0062501, -0.0090324, -0.0062365, -0.0015973, 0.0017047
1: -0.0054497, -0.0047008, -0.0054852, -0.0046970, -0.0004503, 0.0004806
2: -0.0016496, 0.0038763, -0.0019113, 0.0039047, -0.0033227, 0.0035461
3: 0.0014090, 0.0021403, 0.0013744, 0.0021440, -0.0004397, 0.0004693
4: 0.0031950, 0.0073247, 0.0031738, 0.0075203, -0.0026502, 0.0024832
5: 0.9963939, 0.9975412, 0.9963880, 0.9975956, -0.0007363, 0.0006899
6: 0.0046104, 0.0056518, 0.0046050, 0.0057012, -0.0006683, 0.0006262
7: -0.0061763, -0.0022898, -0.0061963, -0.0021057, -0.0024941, 0.0023369
8: -0.0074107, -0.0043858, -0.0075540, -0.0043703, -0.0018188, 0.0019412
9: -0.0036313, -0.0033704, -0.0036327, -0.0033580, -0.0001675, 0.0001569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003490, upper bound: 0.0003767
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003874
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090421, -0.0063423, -0.0090250, -0.0062368, -0.0016203, 0.0015187
1: -0.0054880, -0.0047268, -0.0054831, -0.0046971, -0.0004568, 0.0004282
2: -0.0019316, 0.0036845, -0.0018960, 0.0039040, -0.0033706, 0.0031592
3: 0.0013717, 0.0021149, 0.0013764, 0.0021439, -0.0004461, 0.0004181
4: 0.0033383, 0.0075354, 0.0031743, 0.0075088, -0.0023610, 0.0025190
5: 0.9964337, 0.9975998, 0.9963882, 0.9975923, -0.0006560, 0.0006999
6: 0.0046465, 0.0057050, 0.0046052, 0.0056983, -0.0005954, 0.0006353
7: -0.0060414, -0.0020914, -0.0061958, -0.0021165, -0.0022220, 0.0023707
8: -0.0075651, -0.0044908, -0.0075456, -0.0043707, -0.0018451, 0.0017294
9: -0.0036223, -0.0033571, -0.0036327, -0.0033587, -0.0001492, 0.0001592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B2_A1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003690
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003824
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0088967, -0.0062388, -0.0090250, -0.0062368, -0.0016021, 0.0017276
1: -0.0054470, -0.0046976, -0.0054831, -0.0046971, -0.0004517, 0.0004871
2: -0.0016292, 0.0038998, -0.0018960, 0.0039040, -0.0033327, 0.0035938
3: 0.0014117, 0.0021434, 0.0013764, 0.0021439, -0.0004410, 0.0004756
4: 0.0031774, 0.0073094, 0.0031743, 0.0075088, -0.0026858, 0.0024906
5: 0.9963890, 0.9975370, 0.9963882, 0.9975923, -0.0007462, 0.0006920
6: 0.0046060, 0.0056480, 0.0046052, 0.0056983, -0.0006773, 0.0006281
7: -0.0061929, -0.0023042, -0.0061958, -0.0021165, -0.0025276, 0.0023440
8: -0.0073995, -0.0043729, -0.0075456, -0.0043707, -0.0018243, 0.0019672
9: -0.0036325, -0.0033713, -0.0036327, -0.0033587, -0.0001697, 0.0001574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_B2_B2_A1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003761
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003875
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090171, -0.0063074, -0.0089715, -0.0062737, -0.0016908, 0.0015992
1: -0.0054809, -0.0047170, -0.0054681, -0.0047074, -0.0004767, 0.0004509
2: -0.0018795, 0.0037572, -0.0017847, 0.0038273, -0.0035173, 0.0033266
3: 0.0013786, 0.0021245, 0.0013911, 0.0021338, -0.0004655, 0.0004402
4: 0.0032840, 0.0074965, 0.0032316, 0.0074257, -0.0024861, 0.0026286
5: 0.9964187, 0.9975891, 0.9964041, 0.9975693, -0.0006907, 0.0007303
6: 0.0046328, 0.0056952, 0.0046196, 0.0056773, -0.0006270, 0.0006629
7: -0.0060925, -0.0021281, -0.0061419, -0.0021948, -0.0023397, 0.0024738
8: -0.0075366, -0.0044510, -0.0074847, -0.0044126, -0.0019254, 0.0018210
9: -0.0036257, -0.0033595, -0.0036290, -0.0033640, -0.0001571, 0.0001661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003224, upper bound: 0.0003390
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003224, upper bound: 0.0003624
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090278, -0.0063067, -0.0090003, -0.0062740, -0.0017272, 0.0015933
1: -0.0054839, -0.0047168, -0.0054762, -0.0047075, -0.0004870, 0.0004492
2: -0.0019018, 0.0037586, -0.0018446, 0.0038266, -0.0035930, 0.0033145
3: 0.0013756, 0.0021247, 0.0013832, 0.0021337, -0.0004755, 0.0004386
4: 0.0032829, 0.0075132, 0.0032321, 0.0074704, -0.0024770, 0.0026852
5: 0.9964183, 0.9975936, 0.9964042, 0.9975817, -0.0006882, 0.0007460
6: 0.0046326, 0.0056994, 0.0046198, 0.0056886, -0.0006247, 0.0006772
7: -0.0060936, -0.0021124, -0.0061413, -0.0021527, -0.0023312, 0.0025271
8: -0.0075488, -0.0044502, -0.0075174, -0.0044131, -0.0019668, 0.0018144
9: -0.0036258, -0.0033585, -0.0036290, -0.0033612, -0.0001565, 0.0001697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003500, upper bound: 0.0003550
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003500, upper bound: 0.0003742
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090019, -0.0062954, -0.0089651, -0.0062740, -0.0016915, 0.0016272
1: -0.0054766, -0.0047136, -0.0054663, -0.0047075, -0.0004769, 0.0004588
2: -0.0018480, 0.0037820, -0.0017714, 0.0038266, -0.0035186, 0.0033850
3: 0.0013827, 0.0021278, 0.0013929, 0.0021337, -0.0004656, 0.0004480
4: 0.0032654, 0.0074729, 0.0032321, 0.0074157, -0.0025297, 0.0026296
5: 0.9964135, 0.9975825, 0.9964042, 0.9975665, -0.0007028, 0.0007306
6: 0.0046282, 0.0056892, 0.0046198, 0.0056748, -0.0006380, 0.0006631
7: -0.0061100, -0.0021503, -0.0061414, -0.0022041, -0.0023808, 0.0024747
8: -0.0075193, -0.0044374, -0.0074774, -0.0044130, -0.0019261, 0.0018530
9: -0.0036269, -0.0033610, -0.0036290, -0.0033646, -0.0001599, 0.0001662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003223, upper bound: 0.0003403
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003223, upper bound: 0.0003635
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090125, -0.0062947, -0.0089940, -0.0062743, -0.0017278, 0.0016251
1: -0.0054796, -0.0047134, -0.0054744, -0.0047076, -0.0004871, 0.0004582
2: -0.0018700, 0.0037835, -0.0018315, 0.0038259, -0.0035941, 0.0033806
3: 0.0013798, 0.0021280, 0.0013849, 0.0021336, -0.0004756, 0.0004474
4: 0.0032643, 0.0074894, 0.0032326, 0.0074606, -0.0025264, 0.0026860
5: 0.9964131, 0.9975870, 0.9964043, 0.9975791, -0.0007019, 0.0007463
6: 0.0046279, 0.0056934, 0.0046199, 0.0056861, -0.0006371, 0.0006774
7: -0.0061111, -0.0021348, -0.0061409, -0.0021619, -0.0023777, 0.0025278
8: -0.0075314, -0.0044366, -0.0075103, -0.0044134, -0.0019674, 0.0018505
9: -0.0036270, -0.0033600, -0.0036290, -0.0033618, -0.0001597, 0.0001697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003500, upper bound: 0.0003569
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003500, upper bound: 0.0003762
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0090431, -0.0063213, -0.0089074, -0.0062500, -0.0016213, 0.0014181
1: -0.0054883, -0.0047209, -0.0054500, -0.0047008, -0.0004571, 0.0003998
2: -0.0019337, 0.0037283, -0.0016513, 0.0038766, -0.0033727, 0.0029499
3: 0.0013714, 0.0021207, 0.0014088, 0.0021403, -0.0004463, 0.0003904
4: 0.0033055, 0.0075370, 0.0031947, 0.0073259, -0.0022046, 0.0025206
5: 0.9964246, 0.9976003, 0.9963938, 0.9975416, -0.0006125, 0.0007003
6: 0.0046383, 0.0057054, 0.0046103, 0.0056522, -0.0005560, 0.0006356
7: -0.0060723, -0.0020900, -0.0061766, -0.0022886, -0.0020748, 0.0023721
8: -0.0075662, -0.0044668, -0.0074116, -0.0043856, -0.0018462, 0.0016148
9: -0.0036244, -0.0033570, -0.0036314, -0.0033703, -0.0001393, 0.0001593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003685
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003834
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0090376, -0.0063216, -0.0088969, -0.0062387, -0.0016534, 0.0014228
1: -0.0054867, -0.0047209, -0.0054470, -0.0046976, -0.0004661, 0.0004011
2: -0.0019223, 0.0037277, -0.0016296, 0.0039002, -0.0034394, 0.0029597
3: 0.0013729, 0.0021206, 0.0014116, 0.0021434, -0.0004551, 0.0003917
4: 0.0033060, 0.0075285, 0.0031771, 0.0073097, -0.0022119, 0.0025704
5: 0.9964248, 0.9975979, 0.9963889, 0.9975370, -0.0006145, 0.0007141
6: 0.0046384, 0.0057032, 0.0046059, 0.0056481, -0.0005578, 0.0006482
7: -0.0060718, -0.0020980, -0.0061931, -0.0023039, -0.0020816, 0.0024190
8: -0.0075600, -0.0044672, -0.0073997, -0.0043728, -0.0018827, 0.0016201
9: -0.0036243, -0.0033575, -0.0036325, -0.0033713, -0.0001398, 0.0001624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003763, upper bound: 0.0003682
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003847, upper bound: 0.0003834
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0090424, -0.0063529, -0.0090359, -0.0063022, -0.0016392, 0.0015672
1: -0.0054880, -0.0047298, -0.0054862, -0.0047155, -0.0004621, 0.0004419
2: -0.0019321, 0.0036624, -0.0019186, 0.0037679, -0.0034099, 0.0032601
3: 0.0013716, 0.0021120, 0.0013734, 0.0021259, -0.0004512, 0.0004314
4: 0.0033548, 0.0075358, 0.0032760, 0.0075257, -0.0024364, 0.0025483
5: 0.9964383, 0.9975999, 0.9964164, 0.9975971, -0.0006769, 0.0007080
6: 0.0046507, 0.0057051, 0.0046308, 0.0057025, -0.0006144, 0.0006426
7: -0.0060259, -0.0020911, -0.0061001, -0.0021006, -0.0022929, 0.0023982
8: -0.0075654, -0.0045029, -0.0075580, -0.0044452, -0.0018666, 0.0017846
9: -0.0036212, -0.0033570, -0.0036262, -0.0033577, -0.0001540, 0.0001610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003663, upper bound: 0.0003640
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003745, upper bound: 0.0003805
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0090369, -0.0063532, -0.0090209, -0.0062922, -0.0016713, 0.0015675
1: -0.0054865, -0.0047299, -0.0054820, -0.0047127, -0.0004712, 0.0004420
2: -0.0019207, 0.0036618, -0.0018874, 0.0037888, -0.0034767, 0.0032608
3: 0.0013731, 0.0021119, 0.0013775, 0.0021287, -0.0004601, 0.0004315
4: 0.0033553, 0.0075273, 0.0032603, 0.0075024, -0.0024369, 0.0025983
5: 0.9964384, 0.9975976, 0.9964120, 0.9975906, -0.0006771, 0.0007219
6: 0.0046508, 0.0057029, 0.0046269, 0.0056967, -0.0006146, 0.0006552
7: -0.0060255, -0.0020991, -0.0061148, -0.0021226, -0.0022934, 0.0024453
8: -0.0075591, -0.0045032, -0.0075409, -0.0044337, -0.0019032, 0.0017850
9: -0.0036212, -0.0033576, -0.0036272, -0.0033591, -0.0001540, 0.0001642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003673, upper bound: 0.0003640
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003759, upper bound: 0.0003805
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0090610, -0.0062860, -0.0089065, -0.0062501, -0.0016316, 0.0014506
1: -0.0054933, -0.0047109, -0.0054497, -0.0047008, -0.0004600, 0.0004090
2: -0.0019709, 0.0038016, -0.0016496, 0.0038763, -0.0033941, 0.0030175
3: 0.0013665, 0.0021304, 0.0014090, 0.0021403, -0.0004492, 0.0003993
4: 0.0032508, 0.0075648, 0.0031950, 0.0073247, -0.0022551, 0.0025366
5: 0.9964094, 0.9976079, 0.9963939, 0.9975412, -0.0006265, 0.0007047
6: 0.0046245, 0.0057124, 0.0046104, 0.0056518, -0.0005687, 0.0006397
7: -0.0061238, -0.0020638, -0.0061763, -0.0022898, -0.0021223, 0.0023872
8: -0.0075866, -0.0044267, -0.0074107, -0.0043858, -0.0018579, 0.0016518
9: -0.0036278, -0.0033552, -0.0036313, -0.0033704, -0.0001425, 0.0001603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003730, upper bound: 0.0003694
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003832, upper bound: 0.0003852
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0090552, -0.0062863, -0.0088967, -0.0062388, -0.0016639, 0.0014552
1: -0.0054917, -0.0047110, -0.0054470, -0.0046976, -0.0004691, 0.0004103
2: -0.0019589, 0.0038010, -0.0016292, 0.0038998, -0.0034613, 0.0030272
3: 0.0013681, 0.0021303, 0.0014117, 0.0021434, -0.0004580, 0.0004006
4: 0.0032512, 0.0075558, 0.0031774, 0.0073094, -0.0022623, 0.0025867
5: 0.9964095, 0.9976054, 0.9963890, 0.9975370, -0.0006285, 0.0007187
6: 0.0046246, 0.0057101, 0.0046060, 0.0056480, -0.0005705, 0.0006523
7: -0.0061234, -0.0020723, -0.0061929, -0.0023042, -0.0021291, 0.0024344
8: -0.0075800, -0.0044270, -0.0073995, -0.0043729, -0.0018947, 0.0016571
9: -0.0036278, -0.0033558, -0.0036325, -0.0033713, -0.0001430, 0.0001635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003725, upper bound: 0.0003693
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003832, upper bound: 0.0003853
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0090602, -0.0063207, -0.0090353, -0.0063024, -0.0016474, 0.0015942
1: -0.0054931, -0.0047207, -0.0054861, -0.0047155, -0.0004645, 0.0004495
2: -0.0019693, 0.0037295, -0.0019174, 0.0037676, -0.0034270, 0.0033162
3: 0.0013667, 0.0021208, 0.0013736, 0.0021259, -0.0004535, 0.0004388
4: 0.0033047, 0.0075636, 0.0032762, 0.0075248, -0.0024783, 0.0025611
5: 0.9964244, 0.9976076, 0.9964164, 0.9975969, -0.0006886, 0.0007116
6: 0.0046381, 0.0057121, 0.0046309, 0.0057023, -0.0006250, 0.0006459
7: -0.0060731, -0.0020650, -0.0060999, -0.0021014, -0.0023324, 0.0024103
8: -0.0075857, -0.0044662, -0.0075573, -0.0044453, -0.0018759, 0.0018153
9: -0.0036244, -0.0033553, -0.0036262, -0.0033577, -0.0001566, 0.0001618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003600, upper bound: 0.0003642
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003721, upper bound: 0.0003816
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0090544, -0.0063210, -0.0090201, -0.0062924, -0.0016798, 0.0015947
1: -0.0054914, -0.0047208, -0.0054818, -0.0047127, -0.0004736, 0.0004496
2: -0.0019572, 0.0037289, -0.0018858, 0.0037885, -0.0034943, 0.0033173
3: 0.0013683, 0.0021208, 0.0013777, 0.0021286, -0.0004624, 0.0004390
4: 0.0033051, 0.0075546, 0.0032606, 0.0075012, -0.0024791, 0.0026114
5: 0.9964244, 0.9976051, 0.9964121, 0.9975903, -0.0006888, 0.0007255
6: 0.0046382, 0.0057098, 0.0046269, 0.0056964, -0.0006252, 0.0006586
7: -0.0060727, -0.0020734, -0.0061146, -0.0021237, -0.0023331, 0.0024577
8: -0.0075791, -0.0044665, -0.0075400, -0.0044339, -0.0019128, 0.0018159
9: -0.0036244, -0.0033558, -0.0036272, -0.0033592, -0.0001567, 0.0001650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003613, upper bound: 0.0003642
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003731, upper bound: 0.0003817
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0090431, -0.0063213, -0.0088825, -0.0061972, -0.0017526, 0.0014914
1: -0.0054883, -0.0047209, -0.0054430, -0.0046859, -0.0004941, 0.0004205
2: -0.0019337, 0.0037283, -0.0015996, 0.0039865, -0.0036459, 0.0031023
3: 0.0013714, 0.0021207, 0.0014156, 0.0021548, -0.0004825, 0.0004105
4: 0.0033055, 0.0075370, 0.0031126, 0.0072873, -0.0023185, 0.0027247
5: 0.9964246, 0.9976003, 0.9963710, 0.9975308, -0.0006441, 0.0007570
6: 0.0046383, 0.0057054, 0.0045896, 0.0056424, -0.0005847, 0.0006871
7: -0.0060723, -0.0020900, -0.0062538, -0.0023250, -0.0021820, 0.0025642
8: -0.0075662, -0.0044668, -0.0073833, -0.0043255, -0.0019958, 0.0016982
9: -0.0036244, -0.0033570, -0.0036366, -0.0033727, -0.0001465, 0.0001722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003730
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003882
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0090376, -0.0063216, -0.0088674, -0.0061904, -0.0017748, 0.0014959
1: -0.0054867, -0.0047209, -0.0054387, -0.0046840, -0.0005004, 0.0004218
2: -0.0019223, 0.0037277, -0.0015681, 0.0040006, -0.0036920, 0.0031119
3: 0.0013729, 0.0021206, 0.0014198, 0.0021567, -0.0004886, 0.0004118
4: 0.0033060, 0.0075285, 0.0031021, 0.0072637, -0.0023256, 0.0027591
5: 0.9964248, 0.9975979, 0.9963681, 0.9975244, -0.0006461, 0.0007666
6: 0.0046384, 0.0057032, 0.0045870, 0.0056365, -0.0005865, 0.0006958
7: -0.0060718, -0.0020980, -0.0062638, -0.0023471, -0.0021887, 0.0025967
8: -0.0075600, -0.0044672, -0.0073661, -0.0043178, -0.0020210, 0.0017034
9: -0.0036243, -0.0033575, -0.0036372, -0.0033742, -0.0001470, 0.0001744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003763, upper bound: 0.0003729
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003847, upper bound: 0.0003882
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0090424, -0.0063529, -0.0090149, -0.0062474, -0.0017701, 0.0016208
1: -0.0054880, -0.0047298, -0.0054803, -0.0047000, -0.0004991, 0.0004570
2: -0.0019321, 0.0036624, -0.0018749, 0.0038821, -0.0036822, 0.0033717
3: 0.0013716, 0.0021120, 0.0013792, 0.0021410, -0.0004873, 0.0004462
4: 0.0033548, 0.0075358, 0.0031906, 0.0074930, -0.0025198, 0.0027518
5: 0.9964383, 0.9975999, 0.9963927, 0.9975880, -0.0007001, 0.0007645
6: 0.0046507, 0.0057051, 0.0046093, 0.0056943, -0.0006355, 0.0006940
7: -0.0060259, -0.0020911, -0.0061804, -0.0021313, -0.0023714, 0.0025898
8: -0.0075654, -0.0045029, -0.0075340, -0.0043827, -0.0020156, 0.0018457
9: -0.0036212, -0.0033570, -0.0036316, -0.0033597, -0.0001592, 0.0001739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003664, upper bound: 0.0003703
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003745, upper bound: 0.0003867
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0090369, -0.0063532, -0.0089949, -0.0062404, -0.0017925, 0.0016220
1: -0.0054865, -0.0047299, -0.0054747, -0.0046980, -0.0005054, 0.0004573
2: -0.0019207, 0.0036618, -0.0018334, 0.0038966, -0.0037288, 0.0033740
3: 0.0013731, 0.0021119, 0.0013847, 0.0021430, -0.0004934, 0.0004465
4: 0.0033553, 0.0075273, 0.0031798, 0.0074621, -0.0025215, 0.0027867
5: 0.9964384, 0.9975976, 0.9963896, 0.9975795, -0.0007006, 0.0007742
6: 0.0046508, 0.0057029, 0.0046066, 0.0056865, -0.0006359, 0.0007028
7: -0.0060255, -0.0020991, -0.0061906, -0.0021605, -0.0023730, 0.0026225
8: -0.0075591, -0.0045032, -0.0075113, -0.0043747, -0.0020411, 0.0018469
9: -0.0036212, -0.0033576, -0.0036323, -0.0033617, -0.0001593, 0.0001761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003675, upper bound: 0.0003702
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003762, upper bound: 0.0003867
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0090610, -0.0062860, -0.0088830, -0.0061973, -0.0017629, 0.0015223
1: -0.0054933, -0.0047109, -0.0054431, -0.0046859, -0.0004970, 0.0004292
2: -0.0019709, 0.0038016, -0.0016005, 0.0039862, -0.0036672, 0.0031667
3: 0.0013665, 0.0021304, 0.0014155, 0.0021548, -0.0004853, 0.0004191
4: 0.0032508, 0.0075648, 0.0031129, 0.0072880, -0.0023666, 0.0027406
5: 0.9964094, 0.9976079, 0.9963710, 0.9975310, -0.0006575, 0.0007614
6: 0.0046245, 0.0057124, 0.0045897, 0.0056426, -0.0005968, 0.0006912
7: -0.0061238, -0.0020638, -0.0062536, -0.0023243, -0.0022273, 0.0025793
8: -0.0075866, -0.0044267, -0.0073838, -0.0043257, -0.0020074, 0.0017335
9: -0.0036278, -0.0033552, -0.0036365, -0.0033727, -0.0001496, 0.0001732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003743
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003831, upper bound: 0.0003913
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0090552, -0.0062863, -0.0088679, -0.0061906, -0.0017853, 0.0015268
1: -0.0054917, -0.0047110, -0.0054389, -0.0046840, -0.0005033, 0.0004305
2: -0.0019589, 0.0038010, -0.0015692, 0.0040002, -0.0037138, 0.0031760
3: 0.0013681, 0.0021303, 0.0014196, 0.0021567, -0.0004915, 0.0004203
4: 0.0032512, 0.0075558, 0.0031023, 0.0072646, -0.0023736, 0.0027755
5: 0.9964095, 0.9976054, 0.9963681, 0.9975245, -0.0006594, 0.0007711
6: 0.0046246, 0.0057101, 0.0045870, 0.0056367, -0.0005986, 0.0006999
7: -0.0061234, -0.0020723, -0.0062635, -0.0023463, -0.0022338, 0.0026120
8: -0.0075800, -0.0044270, -0.0073667, -0.0043180, -0.0020330, 0.0017386
9: -0.0036278, -0.0033558, -0.0036372, -0.0033742, -0.0001500, 0.0001754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003725, upper bound: 0.0003743
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003833, upper bound: 0.0003913
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0090602, -0.0063207, -0.0090134, -0.0062475, -0.0017783, 0.0016488
1: -0.0054931, -0.0047207, -0.0054799, -0.0047001, -0.0005014, 0.0004649
2: -0.0019693, 0.0037295, -0.0018719, 0.0038817, -0.0036992, 0.0034298
3: 0.0013667, 0.0021208, 0.0013796, 0.0021410, -0.0004895, 0.0004539
4: 0.0033047, 0.0075636, 0.0031909, 0.0074908, -0.0025632, 0.0027645
5: 0.9964244, 0.9976076, 0.9963928, 0.9975874, -0.0007121, 0.0007681
6: 0.0046381, 0.0057121, 0.0046094, 0.0056937, -0.0006464, 0.0006972
7: -0.0060731, -0.0020650, -0.0061801, -0.0021334, -0.0024123, 0.0026017
8: -0.0075857, -0.0044662, -0.0075324, -0.0043829, -0.0020249, 0.0018775
9: -0.0036244, -0.0033553, -0.0036316, -0.0033599, -0.0001620, 0.0001747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003600, upper bound: 0.0003715
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003724, upper bound: 0.0003896
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0090544, -0.0063210, -0.0089933, -0.0062405, -0.0018009, 0.0016500
1: -0.0054914, -0.0047208, -0.0054742, -0.0046981, -0.0005078, 0.0004652
2: -0.0019572, 0.0037289, -0.0018301, 0.0038963, -0.0037463, 0.0034323
3: 0.0013683, 0.0021208, 0.0013851, 0.0021429, -0.0004958, 0.0004542
4: 0.0033051, 0.0075546, 0.0031800, 0.0074596, -0.0025651, 0.0027998
5: 0.9964244, 0.9976051, 0.9963897, 0.9975787, -0.0007127, 0.0007779
6: 0.0046382, 0.0057098, 0.0046066, 0.0056859, -0.0006469, 0.0007061
7: -0.0060727, -0.0020734, -0.0061904, -0.0021629, -0.0024140, 0.0026349
8: -0.0075791, -0.0044665, -0.0075095, -0.0043749, -0.0020507, 0.0018788
9: -0.0036244, -0.0033558, -0.0036323, -0.0033619, -0.0001621, 0.0001769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003615, upper bound: 0.0003714
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003737, upper bound: 0.0003896
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0088713, -0.0062506, -0.0089463, -0.0062100, -0.0013936, 0.0014417
1: -0.0054398, -0.0047009, -0.0054610, -0.0046895, -0.0003929, 0.0004065
2: -0.0015762, 0.0038752, -0.0017323, 0.0039598, -0.0028991, 0.0029991
3: 0.0014187, 0.0021401, 0.0013981, 0.0021513, -0.0003836, 0.0003969
4: 0.0031958, 0.0072698, 0.0031326, 0.0073864, -0.0022413, 0.0021666
5: 0.9963942, 0.9975260, 0.9963765, 0.9975585, -0.0006227, 0.0006019
6: 0.0046106, 0.0056380, 0.0045947, 0.0056674, -0.0005652, 0.0005464
7: -0.0061756, -0.0023414, -0.0062351, -0.0022317, -0.0021093, 0.0020390
8: -0.0073705, -0.0043864, -0.0074560, -0.0043401, -0.0015869, 0.0016417
9: -0.0036313, -0.0033738, -0.0036353, -0.0033665, -0.0001416, 0.0001369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004316, upper bound: 0.0004303
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004360, upper bound: 0.0004305
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088608, -0.0062393, -0.0089391, -0.0062103, -0.0013944, 0.0014770
1: -0.0054368, -0.0046978, -0.0054589, -0.0046896, -0.0003931, 0.0004164
2: -0.0015544, 0.0038988, -0.0017172, 0.0039592, -0.0029006, 0.0030726
3: 0.0014216, 0.0021432, 0.0014001, 0.0021512, -0.0003838, 0.0004066
4: 0.0031781, 0.0072535, 0.0031330, 0.0073752, -0.0022962, 0.0021677
5: 0.9963892, 0.9975215, 0.9963767, 0.9975553, -0.0006380, 0.0006023
6: 0.0046062, 0.0056339, 0.0045948, 0.0056646, -0.0005791, 0.0005467
7: -0.0061921, -0.0023568, -0.0062346, -0.0022423, -0.0021610, 0.0020400
8: -0.0073586, -0.0043735, -0.0074477, -0.0043404, -0.0015878, 0.0016819
9: -0.0036324, -0.0033749, -0.0036353, -0.0033672, -0.0001451, 0.0001370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004315, upper bound: 0.0004321
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004360, upper bound: 0.0004322
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0089995, -0.0063030, -0.0089454, -0.0062393, -0.0015323, 0.0014626
1: -0.0054759, -0.0047157, -0.0054607, -0.0046978, -0.0004320, 0.0004123
2: -0.0018429, 0.0037664, -0.0017304, 0.0038988, -0.0031875, 0.0030424
3: 0.0013834, 0.0021257, 0.0013983, 0.0021432, -0.0004218, 0.0004026
4: 0.0032771, 0.0074691, 0.0031781, 0.0073850, -0.0022737, 0.0023821
5: 0.9964167, 0.9975814, 0.9963892, 0.9975581, -0.0006317, 0.0006618
6: 0.0046311, 0.0056883, 0.0046061, 0.0056671, -0.0005734, 0.0006007
7: -0.0060990, -0.0021539, -0.0061922, -0.0022330, -0.0021398, 0.0022418
8: -0.0075165, -0.0044460, -0.0074549, -0.0043735, -0.0017448, 0.0016654
9: -0.0036262, -0.0033612, -0.0036324, -0.0033666, -0.0001437, 0.0001505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004393, upper bound: 0.0004272
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004305
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0089848, -0.0062929, -0.0089382, -0.0062396, -0.0015285, 0.0014988
1: -0.0054718, -0.0047129, -0.0054587, -0.0046978, -0.0004310, 0.0004226
2: -0.0018125, 0.0037873, -0.0017154, 0.0038982, -0.0031797, 0.0031177
3: 0.0013874, 0.0021285, 0.0014003, 0.0021432, -0.0004208, 0.0004126
4: 0.0032615, 0.0074464, 0.0031786, 0.0073738, -0.0023300, 0.0023763
5: 0.9964124, 0.9975750, 0.9963893, 0.9975549, -0.0006473, 0.0006602
6: 0.0046272, 0.0056825, 0.0046063, 0.0056642, -0.0005876, 0.0005993
7: -0.0061137, -0.0021753, -0.0061918, -0.0022435, -0.0021928, 0.0022364
8: -0.0074999, -0.0044345, -0.0074467, -0.0043738, -0.0017406, 0.0017066
9: -0.0036271, -0.0033627, -0.0036324, -0.0033673, -0.0001472, 0.0001502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004393, upper bound: 0.0004296
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004322
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0088769, -0.0062263, -0.0089456, -0.0062102, -0.0014002, 0.0014753
1: -0.0054414, -0.0046941, -0.0054607, -0.0046895, -0.0003948, 0.0004159
2: -0.0015878, 0.0039259, -0.0017308, 0.0039595, -0.0029127, 0.0030689
3: 0.0014172, 0.0021468, 0.0013983, 0.0021513, -0.0003854, 0.0004061
4: 0.0031579, 0.0072785, 0.0031328, 0.0073853, -0.0022935, 0.0021768
5: 0.9963836, 0.9975284, 0.9963766, 0.9975581, -0.0006372, 0.0006048
6: 0.0046010, 0.0056402, 0.0045947, 0.0056671, -0.0005784, 0.0005489
7: -0.0062112, -0.0023333, -0.0062348, -0.0022327, -0.0021584, 0.0020486
8: -0.0073769, -0.0043587, -0.0074551, -0.0043403, -0.0015944, 0.0016799
9: -0.0036337, -0.0033733, -0.0036353, -0.0033665, -0.0001449, 0.0001376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004315, upper bound: 0.0004379
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004360, upper bound: 0.0004382
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0088683, -0.0062124, -0.0089386, -0.0062105, -0.0014020, 0.0015104
1: -0.0054390, -0.0046902, -0.0054588, -0.0046896, -0.0003953, 0.0004258
2: -0.0015701, 0.0039548, -0.0017162, 0.0039588, -0.0029165, 0.0031420
3: 0.0014195, 0.0021506, 0.0014002, 0.0021512, -0.0003860, 0.0004158
4: 0.0031363, 0.0072652, 0.0031333, 0.0073744, -0.0023481, 0.0021796
5: 0.9963776, 0.9975248, 0.9963768, 0.9975551, -0.0006524, 0.0006056
6: 0.0045956, 0.0056369, 0.0045948, 0.0056644, -0.0005922, 0.0005497
7: -0.0062315, -0.0023457, -0.0062344, -0.0022430, -0.0022098, 0.0020513
8: -0.0073672, -0.0043429, -0.0074472, -0.0043406, -0.0015965, 0.0017199
9: -0.0036351, -0.0033741, -0.0036352, -0.0033672, -0.0001484, 0.0001377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004317, upper bound: 0.0004392
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004360, upper bound: 0.0004393
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0090072, -0.0062805, -0.0089447, -0.0062395, -0.0015406, 0.0014971
1: -0.0054781, -0.0047094, -0.0054605, -0.0046978, -0.0004343, 0.0004221
2: -0.0018590, 0.0038132, -0.0017289, 0.0038985, -0.0032047, 0.0031142
3: 0.0013813, 0.0021319, 0.0013985, 0.0021432, -0.0004241, 0.0004121
4: 0.0032421, 0.0074811, 0.0031784, 0.0073839, -0.0023274, 0.0023950
5: 0.9964070, 0.9975847, 0.9963893, 0.9975578, -0.0006466, 0.0006654
6: 0.0046223, 0.0056913, 0.0046062, 0.0056668, -0.0005869, 0.0006040
7: -0.0061320, -0.0021426, -0.0061919, -0.0022341, -0.0021903, 0.0022540
8: -0.0075253, -0.0044204, -0.0074541, -0.0043737, -0.0017543, 0.0017047
9: -0.0036284, -0.0033605, -0.0036324, -0.0033666, -0.0001471, 0.0001513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004341
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004381
time: 1.07 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.56 seconds
IS_A1_B1_B1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003477, upper bound: 0.0003672
IS_A1_B1_B1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003782
IS_A1_B1_B1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003477, upper bound: 0.0003766
IS_A1_B1_B1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003844
IS_A1_B1_B1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003474, upper bound: 0.0003667
IS_A1_B1_B1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003781
IS_A1_B1_B1_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003474, upper bound: 0.0003764
IS_A1_B1_B1_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003848
IS_A1_B1_B1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003216, upper bound: 0.0003387
IS_A1_B1_B1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003216, upper bound: 0.0003664
IS_A1_B1_B1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003482, upper bound: 0.0003506
IS_A1_B1_B1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003482, upper bound: 0.0003746
IS_A1_B1_B1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003215, upper bound: 0.0003402
IS_A1_B1_B1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003215, upper bound: 0.0003674
IS_A1_B1_B1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003482, upper bound: 0.0003527
IS_A1_B1_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003482, upper bound: 0.0003759
IS_A1_B1_B1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003490, upper bound: 0.0003638
IS_A1_B1_B1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003766
IS_A1_B1_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003490, upper bound: 0.0003731
IS_A1_B1_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003832
IS_A1_B1_B1_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003630
IS_A1_B1_B1_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003764
IS_A1_B1_B1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003726
IS_A1_B1_B1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003833
IS_A1_B1_B1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003221, upper bound: 0.0003328
IS_A1_B1_B1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003221, upper bound: 0.0003601
IS_A1_B1_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003499, upper bound: 0.0003482
IS_A1_B1_B1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003499, upper bound: 0.0003721
IS_A1_B1_B1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003221, upper bound: 0.0003341
IS_A1_B1_B1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003221, upper bound: 0.0003613
IS_A1_B1_B1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003499, upper bound: 0.0003500
IS_A1_B1_B1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003499, upper bound: 0.0003732
IS_A1_B1_B2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003477, upper bound: 0.0003722
IS_A1_B1_B2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003837
IS_A1_B1_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003477, upper bound: 0.0003791
IS_A1_B1_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003881
IS_A1_B1_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003474, upper bound: 0.0003723
IS_A1_B1_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003840
IS_A1_B1_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003474, upper bound: 0.0003789
IS_A1_B1_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003678, upper bound: 0.0003884
IS_A1_B1_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003218, upper bound: 0.0003440
IS_A1_B1_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003218, upper bound: 0.0003677
IS_A1_B1_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003484, upper bound: 0.0003561
IS_A1_B1_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003484, upper bound: 0.0003759
IS_A1_B1_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003216, upper bound: 0.0003455
IS_A1_B1_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003216, upper bound: 0.0003687
IS_A1_B1_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003484, upper bound: 0.0003582
IS_A1_B1_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003484, upper bound: 0.0003777
IS_A1_B1_B2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003490, upper bound: 0.0003693
IS_A1_B1_B2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003824
IS_A1_B1_B2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003490, upper bound: 0.0003767
IS_A1_B1_B2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003874
IS_A1_B1_B2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003690
IS_A1_B1_B2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003824
IS_A1_B1_B2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003761
IS_A1_B1_B2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003711, upper bound: 0.0003875
IS_A1_B1_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003224, upper bound: 0.0003390
IS_A1_B1_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003224, upper bound: 0.0003624
IS_A1_B1_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003500, upper bound: 0.0003550
IS_A1_B1_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003500, upper bound: 0.0003742
IS_A1_B1_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003223, upper bound: 0.0003403
IS_A1_B1_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003223, upper bound: 0.0003635
IS_A1_B1_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003500, upper bound: 0.0003569
IS_A1_B1_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003500, upper bound: 0.0003762
IS_A1_B2_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003685
IS_A1_B2_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003834
IS_A1_B2_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003763, upper bound: 0.0003682
IS_A1_B2_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003847, upper bound: 0.0003834
IS_A1_B2_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003663, upper bound: 0.0003640
IS_A1_B2_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003745, upper bound: 0.0003805
IS_A1_B2_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003673, upper bound: 0.0003640
IS_A1_B2_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003759, upper bound: 0.0003805
IS_A1_B2_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003730, upper bound: 0.0003694
IS_A1_B2_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003832, upper bound: 0.0003852
IS_A1_B2_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003725, upper bound: 0.0003693
IS_A1_B2_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003832, upper bound: 0.0003853
IS_A1_B2_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003600, upper bound: 0.0003642
IS_A1_B2_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003721, upper bound: 0.0003816
IS_A1_B2_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003613, upper bound: 0.0003642
IS_A1_B2_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003731, upper bound: 0.0003817
IS_A1_B2_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003730
IS_A1_B2_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003882
IS_A1_B2_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003763, upper bound: 0.0003729
IS_A1_B2_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003847, upper bound: 0.0003882
IS_A1_B2_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003664, upper bound: 0.0003703
IS_A1_B2_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003745, upper bound: 0.0003867
IS_A1_B2_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003675, upper bound: 0.0003702
IS_A1_B2_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003762, upper bound: 0.0003867
IS_A1_B2_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003743
IS_A1_B2_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003831, upper bound: 0.0003913
IS_A1_B2_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003725, upper bound: 0.0003743
IS_A1_B2_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003833, upper bound: 0.0003913
IS_A1_B2_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003600, upper bound: 0.0003715
IS_A1_B2_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003724, upper bound: 0.0003896
IS_A1_B2_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003615, upper bound: 0.0003714
IS_A1_B2_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0003737, upper bound: 0.0003896
IS_A1_B2_A2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004316, upper bound: 0.0004303
IS_A1_B2_A2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004360, upper bound: 0.0004305
IS_A1_B2_A2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004315, upper bound: 0.0004321
IS_A1_B2_A2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004360, upper bound: 0.0004322
IS_A1_B2_A2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004393, upper bound: 0.0004272
IS_A1_B2_A2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004305
IS_A1_B2_A2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004393, upper bound: 0.0004296
IS_A1_B2_A2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004322
IS_A1_B2_A2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004315, upper bound: 0.0004379
IS_A1_B2_A2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004360, upper bound: 0.0004382
IS_A1_B2_A2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004317, upper bound: 0.0004392
IS_A1_B2_A2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004360, upper bound: 0.0004393
IS_A1_B2_A2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004341
IS_A1_B2_A2_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.56
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004381
IS_A1_B2_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004474
IS_A1_B2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004423, upper bound: 0.0004401
IS_A1_B2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004401
IS_A1_B2_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004401
IS_A1_B2_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004401
IS_A1_B2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004423, upper bound: 0.0004486
IS_A1_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004486
IS_A1_B2_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004486
IS_A1_B2_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004475, upper bound: 0.0004486
IS_A2_B1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003868
IS_A2_B1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003930
IS_A2_B1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003868
IS_A2_B1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003802, upper bound: 0.0003934
IS_A2_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003702, upper bound: 0.0003664
IS_A2_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003866, upper bound: 0.0003745
IS_A2_B1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003702, upper bound: 0.0003675
IS_A2_B1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003866, upper bound: 0.0003763
IS_A2_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003854
IS_A2_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003844, upper bound: 0.0003920
IS_A2_B1_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003843, upper bound: 0.0003852
IS_A2_B1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003843, upper bound: 0.0003921
IS_A2_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003714, upper bound: 0.0003602
IS_A2_B1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003725
IS_A2_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003714, upper bound: 0.0003617
IS_A2_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003895, upper bound: 0.0003738
IS_A2_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003932
IS_A2_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003987
IS_A2_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003937
IS_A2_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003850, upper bound: 0.0003994
IS_A2_B1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003735, upper bound: 0.0003705
IS_A2_B1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003897, upper bound: 0.0003810
IS_A2_B1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003735, upper bound: 0.0003723
IS_A2_B1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003897, upper bound: 0.0003826
IS_A2_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003886, upper bound: 0.0003924
IS_A2_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003886, upper bound: 0.0003983
IS_A2_B1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003885, upper bound: 0.0003930
IS_A2_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003885, upper bound: 0.0003990
IS_A2_B1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003754, upper bound: 0.0003664
IS_A2_B1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003922, upper bound: 0.0003797
IS_A2_B1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003754, upper bound: 0.0003681
IS_A2_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003922, upper bound: 0.0003814
IS_A2_B2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003967, upper bound: 0.0003924
IS_A2_B2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003971, upper bound: 0.0003923
IS_A2_B2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003841, upper bound: 0.0003895
IS_A2_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003861, upper bound: 0.0003895
IS_A2_B2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003962, upper bound: 0.0003940
IS_A2_B2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003964, upper bound: 0.0003941
IS_A2_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003827, upper bound: 0.0003903
IS_A2_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003847, upper bound: 0.0003903
IS_A2_B2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004423
IS_A2_B2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004437
IS_A2_B2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004432, upper bound: 0.0004474
IS_A2_B2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004429, upper bound: 0.0004474
IS_A2_B2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004493, upper bound: 0.0004423
IS_A2_B2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004493, upper bound: 0.0004436
IS_A2_B2_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004498, upper bound: 0.0004474
IS_A2_B2_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004493, upper bound: 0.0004474
IS_A2_B2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004000, upper bound: 0.0003979
IS_A2_B2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004003, upper bound: 0.0003979
IS_A2_B2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003892, upper bound: 0.0003961
IS_A2_B2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003910, upper bound: 0.0003961
IS_A2_B2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003995, upper bound: 0.0004001
IS_A2_B2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003999, upper bound: 0.0004001
IS_A2_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003880, upper bound: 0.0003980
IS_A2_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0003901, upper bound: 0.0003980
IS_A2_B2_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004391
IS_A2_B2_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004398
IS_A2_B2_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004497, upper bound: 0.0004391
IS_A2_B2_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004496, upper bound: 0.0004398
IS_A2_B2_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004467
IS_A2_B2_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004458, upper bound: 0.0004474
IS_A2_B2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004497, upper bound: 0.0004467
IS_A2_B2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 5, lower bound: -0.0004497, upper bound: 0.0004474

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.09 + 598.31 = 601.41 seconds
