## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.045817377500000006


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9380678, 0.9917637, 0.9380678, 0.9917637, -0.0536959, 0.0536959)
1: (-0.0115945, -0.0026022, -0.0115945, -0.0026022, -0.0089923, 0.0089923)
2: (0.0075203, 0.0204642, 0.0075203, 0.0204642, -0.0129439, 0.0129439)
3: (-0.0090238, 0.0055741, -0.0090238, 0.0055741, -0.0145979, 0.0145979)
4: (-0.0041402, 0.0057734, -0.0041402, 0.0057734, -0.0099136, 0.0099136)
5: (0.0084180, 0.0549622, 0.0084180, 0.0549622, -0.0465442, 0.0465442)
6: (-0.0082169, 0.0051646, -0.0082169, 0.0051646, -0.0133815, 0.0133815)
7: (-0.0184287, -0.0037830, -0.0184287, -0.0037830, -0.0146457, 0.0146457)
8: (-0.0084818, 0.0135686, -0.0084818, 0.0135686, -0.0220504, 0.0220504)
9: (-0.0001674, 0.0124900, -0.0001674, 0.0124900, -0.0126574, 0.0126574)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 2.26 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0495323, upper bound: 0.0495323

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0490689, upper bound: 0.0489022
time: 1.37 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0489050, upper bound: 0.0489050
time: 1.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.81
Output dim: 0, lower bound: -0.0490689, upper bound: 0.0489022
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.81
Output dim: 0, lower bound: -0.0489050, upper bound: 0.0489050

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9386791, 0.9914124, 0.9380678, 0.9917637, -0.0530846, 0.0533446
1: -0.0114181, -0.0026152, -0.0115945, -0.0026022, -0.0088159, 0.0089793
2: 0.0079842, 0.0203440, 0.0075203, 0.0204642, -0.0124800, 0.0128237
3: -0.0090123, 0.0052324, -0.0090238, 0.0055741, -0.0145864, 0.0142562
4: -0.0039524, 0.0057547, -0.0041402, 0.0057734, -0.0097258, 0.0098949
5: 0.0090015, 0.0544196, 0.0084180, 0.0549622, -0.0459607, 0.0460016
6: -0.0080944, 0.0050113, -0.0082169, 0.0051646, -0.0132590, 0.0132282
7: -0.0182496, -0.0038665, -0.0184287, -0.0037830, -0.0144666, 0.0145623
8: -0.0083911, 0.0133192, -0.0084818, 0.0135686, -0.0219598, 0.0218010
9: -0.0000753, 0.0123065, -0.0001674, 0.0124900, -0.0125653, 0.0124739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0488757, upper bound: 0.0488757
time: 1.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0488757, upper bound: 0.0488757
time: 1.25 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9191378, 0.9913099, 0.9382010, 0.9916694, -0.0725316, 0.0531089
1: -0.0170580, -0.0021990, -0.0115561, -0.0026050, -0.0144530, 0.0093571
2: 0.0056050, 0.0241867, 0.0076448, 0.0204380, -0.0148330, 0.0165419
3: -0.0093804, 0.0161566, -0.0090213, 0.0054996, -0.0148800, 0.0251779
4: -0.0099582, 0.0063534, -0.0040993, 0.0057693, -0.0157275, 0.0104527
5: 0.0076958, 0.0717655, 0.0085746, 0.0548440, -0.0471481, 0.0631909
6: -0.0120089, 0.0099125, -0.0081902, 0.0051312, -0.0171401, 0.0181027
7: -0.0239759, -0.0011980, -0.0183897, -0.0038012, -0.0201748, 0.0171916
8: -0.0112896, 0.0212935, -0.0084620, 0.0135143, -0.0248039, 0.0297556
9: -0.0030198, 0.0181714, -0.0001474, 0.0124500, -0.0154698, 0.0183187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482741, upper bound: 0.0485651
time: 1.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483157, upper bound: 0.0483157
time: 1.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.92 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 0, lower bound: -0.0488757, upper bound: 0.0488757
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 0, lower bound: -0.0488757, upper bound: 0.0488757
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 0, lower bound: -0.0482741, upper bound: 0.0485651
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 0, lower bound: -0.0483157, upper bound: 0.0483157

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9386791, 0.9914124, 0.9386791, 0.9914124, -0.0527333, 0.0527333
1: -0.0114181, -0.0026152, -0.0114181, -0.0026152, -0.0088029, 0.0088029
2: 0.0079842, 0.0203440, 0.0079842, 0.0203440, -0.0123598, 0.0123598
3: -0.0090123, 0.0052324, -0.0090123, 0.0052324, -0.0142447, 0.0142447
4: -0.0039524, 0.0057547, -0.0039524, 0.0057547, -0.0097070, 0.0097070
5: 0.0090015, 0.0544196, 0.0090015, 0.0544196, -0.0454181, 0.0454181
6: -0.0080944, 0.0050113, -0.0080944, 0.0050113, -0.0131057, 0.0131057
7: -0.0182496, -0.0038665, -0.0182496, -0.0038665, -0.0143831, 0.0143831
8: -0.0083911, 0.0133192, -0.0083911, 0.0133192, -0.0217103, 0.0217103
9: -0.0000753, 0.0123065, -0.0000753, 0.0123065, -0.0123818, 0.0123818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0487333, upper bound: 0.0482659
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484891, upper bound: 0.0483261
time: 1.41 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9386791, 0.9914124, 0.9191378, 0.9913099, -0.0526308, 0.0722746
1: -0.0114181, -0.0026152, -0.0170580, -0.0021990, -0.0092191, 0.0144428
2: 0.0079842, 0.0203440, 0.0056050, 0.0241867, -0.0162025, 0.0147390
3: -0.0090123, 0.0052324, -0.0093804, 0.0161566, -0.0251689, 0.0146128
4: -0.0039524, 0.0057547, -0.0099582, 0.0063534, -0.0103057, 0.0157129
5: 0.0090015, 0.0544196, 0.0076958, 0.0717655, -0.0627640, 0.0467238
6: -0.0080944, 0.0050113, -0.0120089, 0.0099125, -0.0180069, 0.0170202
7: -0.0182496, -0.0038665, -0.0239759, -0.0011980, -0.0170515, 0.0201095
8: -0.0083911, 0.0133192, -0.0112896, 0.0212935, -0.0296847, 0.0246088
9: -0.0000753, 0.0123065, -0.0030198, 0.0181714, -0.0182467, 0.0153263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0487333, upper bound: 0.0482659
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484891, upper bound: 0.0483261
time: 1.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9191535, 0.9912877, 0.9384945, 0.9912630, -0.0721095, 0.0527932
1: -0.0170535, -0.0021993, -0.0114714, -0.0026113, -0.0144422, 0.0092721
2: 0.0056072, 0.0241836, 0.0081814, 0.0203803, -0.0147731, 0.0160021
3: -0.0093801, 0.0161478, -0.0090158, 0.0053356, -0.0147157, 0.0251635
4: -0.0099533, 0.0063529, -0.0040091, 0.0057603, -0.0157136, 0.0103620
5: 0.0076991, 0.0717514, 0.0092495, 0.0545835, -0.0468844, 0.0625019
6: -0.0120058, 0.0099085, -0.0081314, 0.0050576, -0.0170633, 0.0180399
7: -0.0239713, -0.0012002, -0.0183037, -0.0038412, -0.0201301, 0.0171035
8: -0.0112872, 0.0212871, -0.0084185, 0.0133945, -0.0246818, 0.0297056
9: -0.0030174, 0.0181666, -0.0001031, 0.0123619, -0.0153793, 0.0182698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482438, upper bound: 0.0485651
time: 1.38 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482438, upper bound: 0.0485651
time: 1.52 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9191734, 0.9912757, 0.9269562, 0.9914325, -0.0722592, 0.0643194
1: -0.0170478, -0.0021997, -0.0148015, -0.0023655, -0.0146822, 0.0126017
2: 0.0056100, 0.0241797, 0.0066959, 0.0226492, -0.0170392, 0.0174837
3: -0.0093797, 0.0161367, -0.0092331, 0.0117858, -0.0211655, 0.0253698
4: -0.0099472, 0.0063523, -0.0075552, 0.0061138, -0.0160610, 0.0139075
5: 0.0077032, 0.0717338, 0.0089678, 0.0648253, -0.0571221, 0.0627660
6: -0.0120018, 0.0099035, -0.0104427, 0.0079515, -0.0199533, 0.0203462
7: -0.0239655, -0.0012029, -0.0216848, -0.0022657, -0.0216998, 0.0204819
8: -0.0112843, 0.0212790, -0.0101299, 0.0181030, -0.0293873, 0.0314089
9: -0.0030144, 0.0181607, -0.0018417, 0.0158248, -0.0188392, 0.0200024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482981, upper bound: 0.0483157
time: 1.36 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482981, upper bound: 0.0483157
time: 1.16 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.81 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -0.0487333, upper bound: 0.0482659
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -0.0484891, upper bound: 0.0483261
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -0.0487333, upper bound: 0.0482659
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -0.0484891, upper bound: 0.0483261
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -0.0482438, upper bound: 0.0485651
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -0.0482438, upper bound: 0.0485651
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -0.0482981, upper bound: 0.0483157
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -0.0482981, upper bound: 0.0483157

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9389619, 0.9909965, 0.9386939, 0.9913889, -0.0524270, 0.0523025
1: -0.0113365, -0.0026212, -0.0114138, -0.0026155, -0.0087210, 0.0087926
2: 0.0083711, 0.0202884, 0.0080151, 0.0203411, -0.0119700, 0.0122733
3: -0.0090070, 0.0050743, -0.0090120, 0.0052241, -0.0142310, 0.0140863
4: -0.0038655, 0.0057460, -0.0039478, 0.0057542, -0.0096197, 0.0096938
5: 0.0096922, 0.0541686, 0.0090403, 0.0544065, -0.0447143, 0.0451282
6: -0.0080378, 0.0049404, -0.0080915, 0.0050076, -0.0130454, 0.0130318
7: -0.0181667, -0.0039051, -0.0182452, -0.0038685, -0.0142982, 0.0143402
8: -0.0083492, 0.0132038, -0.0083889, 0.0133131, -0.0216623, 0.0215927
9: -0.0000327, 0.0122216, -0.0000731, 0.0123020, -0.0123347, 0.0122947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484801, upper bound: 0.0483332
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0486065, upper bound: 0.0483244
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9274435, 0.9911716, 0.9387137, 0.9913808, -0.0639373, 0.0524579
1: -0.0146608, -0.0023759, -0.0114082, -0.0026160, -0.0120449, 0.0090323
2: 0.0067639, 0.0225534, 0.0080258, 0.0203372, -0.0135733, 0.0145276
3: -0.0092239, 0.0115134, -0.0090116, 0.0052131, -0.0144370, 0.0205250
4: -0.0074055, 0.0060989, -0.0039418, 0.0057536, -0.0131591, 0.0100407
5: 0.0094014, 0.0643927, 0.0090538, 0.0543890, -0.0449876, 0.0553389
6: -0.0103451, 0.0078293, -0.0080875, 0.0050026, -0.0153477, 0.0159168
7: -0.0215420, -0.0023322, -0.0182395, -0.0038712, -0.0176709, 0.0159073
8: -0.0100576, 0.0179041, -0.0083860, 0.0133051, -0.0233628, 0.0262901
9: -0.0017683, 0.0156786, -0.0000701, 0.0122962, -0.0140644, 0.0157487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482560, upper bound: 0.0483921
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483822, upper bound: 0.0483822
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9389619, 0.9909965, 0.9191535, 0.9912877, -0.0523258, 0.0718430
1: -0.0113365, -0.0026212, -0.0170535, -0.0021993, -0.0091372, 0.0144322
2: 0.0083711, 0.0202884, 0.0056072, 0.0241836, -0.0158125, 0.0146812
3: -0.0090070, 0.0050743, -0.0093801, 0.0161478, -0.0251547, 0.0144544
4: -0.0038655, 0.0057460, -0.0099533, 0.0063529, -0.0102183, 0.0156993
5: 0.0096922, 0.0541686, 0.0076991, 0.0717514, -0.0620593, 0.0464695
6: -0.0080378, 0.0049404, -0.0120058, 0.0099085, -0.0179463, 0.0169461
7: -0.0181667, -0.0039051, -0.0239713, -0.0012002, -0.0169665, 0.0200663
8: -0.0083492, 0.0132038, -0.0112872, 0.0212871, -0.0296363, 0.0244910
9: -0.0000327, 0.0122216, -0.0030174, 0.0181666, -0.0181994, 0.0152390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0485207, upper bound: 0.0479187
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0485050, upper bound: 0.0480361
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9274435, 0.9911716, 0.9191734, 0.9912757, -0.0638322, 0.0719982
1: -0.0146608, -0.0023759, -0.0170478, -0.0021997, -0.0124611, 0.0146719
2: 0.0067639, 0.0225534, 0.0056100, 0.0241797, -0.0174157, 0.0169434
3: -0.0092239, 0.0115134, -0.0093797, 0.0161367, -0.0253606, 0.0208931
4: -0.0074055, 0.0060989, -0.0099472, 0.0063523, -0.0137577, 0.0160461
5: 0.0094014, 0.0643927, 0.0077032, 0.0717338, -0.0623324, 0.0566895
6: -0.0103451, 0.0078293, -0.0120018, 0.0099035, -0.0202486, 0.0198311
7: -0.0215420, -0.0023322, -0.0239655, -0.0012029, -0.0203391, 0.0216333
8: -0.0100576, 0.0179041, -0.0112843, 0.0212790, -0.0313366, 0.0291884
9: -0.0017683, 0.0156786, -0.0030144, 0.0181607, -0.0199290, 0.0186930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482896, upper bound: 0.0479741
time: 1.48 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482756, upper bound: 0.0480973
time: 1.57 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9191535, 0.9912877, 0.9389619, 0.9909965, -0.0718430, 0.0523258
1: -0.0170535, -0.0021993, -0.0113365, -0.0026212, -0.0144322, 0.0091372
2: 0.0056072, 0.0241836, 0.0083711, 0.0202884, -0.0146812, 0.0158125
3: -0.0093801, 0.0161478, -0.0090070, 0.0050743, -0.0144544, 0.0251547
4: -0.0099533, 0.0063529, -0.0038655, 0.0057460, -0.0156993, 0.0102183
5: 0.0076991, 0.0717514, 0.0096922, 0.0541686, -0.0464695, 0.0620593
6: -0.0120058, 0.0099085, -0.0080378, 0.0049404, -0.0169461, 0.0179463
7: -0.0239713, -0.0012002, -0.0181667, -0.0039051, -0.0200663, 0.0169665
8: -0.0112872, 0.0212871, -0.0083492, 0.0132038, -0.0244910, 0.0296363
9: -0.0030174, 0.0181666, -0.0000327, 0.0122216, -0.0152390, 0.0181994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483500
time: 1.49 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480155, upper bound: 0.0483270
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9191535, 0.9912877, 0.9194407, 0.9909059, -0.0717524, 0.0718470
1: -0.0170535, -0.0021993, -0.0169706, -0.0022054, -0.0148481, 0.0147712
2: 0.0056072, 0.0241836, 0.0056473, 0.0241271, -0.0185199, 0.0185363
3: -0.0093801, 0.0161478, -0.0093747, 0.0159872, -0.0253672, 0.0255224
4: -0.0099533, 0.0063529, -0.0098650, 0.0063441, -0.0162974, 0.0162179
5: 0.0076991, 0.0717514, 0.0077583, 0.0714964, -0.0637973, 0.0639931
6: -0.0120058, 0.0099085, -0.0119482, 0.0098364, -0.0218422, 0.0218567
7: -0.0239713, -0.0012002, -0.0238871, -0.0012394, -0.0227319, 0.0226869
8: -0.0112872, 0.0212871, -0.0112446, 0.0211699, -0.0324571, 0.0325317
9: -0.0030174, 0.0181666, -0.0029741, 0.0180804, -0.0210978, 0.0211408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0482085
time: 1.50 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480155, upper bound: 0.0483270
time: 1.48 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9191734, 0.9912757, 0.9274435, 0.9911716, -0.0719982, 0.0638322
1: -0.0170478, -0.0021997, -0.0146608, -0.0023759, -0.0146719, 0.0124611
2: 0.0056100, 0.0241797, 0.0067639, 0.0225534, -0.0169434, 0.0174157
3: -0.0093797, 0.0161367, -0.0092239, 0.0115134, -0.0208931, 0.0253606
4: -0.0099472, 0.0063523, -0.0074055, 0.0060989, -0.0160461, 0.0137577
5: 0.0077032, 0.0717338, 0.0094014, 0.0643927, -0.0566895, 0.0623324
6: -0.0120018, 0.0099035, -0.0103451, 0.0078293, -0.0198311, 0.0202486
7: -0.0239655, -0.0012029, -0.0215420, -0.0023322, -0.0216333, 0.0203391
8: -0.0112843, 0.0212790, -0.0100576, 0.0179041, -0.0291884, 0.0313366
9: -0.0030144, 0.0181607, -0.0017683, 0.0156786, -0.0186930, 0.0199290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479431, upper bound: 0.0481072
time: 1.46 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480678, upper bound: 0.0480843
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9191734, 0.9912757, 0.9087126, 0.9910525, -0.0718791, 0.0825631
1: -0.0170478, -0.0021997, -0.0200669, -0.0019769, -0.0150708, 0.0178672
2: 0.0056100, 0.0241797, 0.0041504, 0.0262367, -0.0206267, 0.0200293
3: -0.0093797, 0.0161367, -0.0095767, 0.0219846, -0.0313643, 0.0257134
4: -0.0099472, 0.0063523, -0.0131622, 0.0066728, -0.0166200, 0.0195145
5: 0.0077032, 0.0717338, 0.0055474, 0.0810193, -0.0733160, 0.0661864
6: -0.0120018, 0.0099035, -0.0140973, 0.0125272, -0.0245290, 0.0240008
7: -0.0239655, -0.0012029, -0.0270309, 0.0002255, -0.0241910, 0.0258280
8: -0.0112843, 0.0212790, -0.0128359, 0.0255478, -0.0368321, 0.0341149
9: -0.0030144, 0.0181607, -0.0045906, 0.0213002, -0.0243147, 0.0227513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480874, upper bound: 0.0479684
time: 1.68 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480678, upper bound: 0.0480843
time: 1.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.55 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0484801, upper bound: 0.0483332
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0486065, upper bound: 0.0483244
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0482560, upper bound: 0.0483921
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0483822, upper bound: 0.0483822
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0485207, upper bound: 0.0479187
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0485050, upper bound: 0.0480361
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0482896, upper bound: 0.0479741
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0482756, upper bound: 0.0480973
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483500
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0480155, upper bound: 0.0483270
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0482085
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0480155, upper bound: 0.0483270
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0479431, upper bound: 0.0481072
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0480678, upper bound: 0.0480843
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0480874, upper bound: 0.0479684
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 0, lower bound: -0.0480678, upper bound: 0.0480843

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9390321, 0.9909468, 0.9479476, 0.9909860, -0.0519539, 0.0429991
1: -0.0113162, -0.0026227, -0.0087430, -0.0028126, -0.0085036, 0.0061203
2: 0.0083809, 0.0202746, 0.0085471, 0.0185214, -0.0101405, 0.0116459
3: -0.0090056, 0.0050350, -0.0088377, 0.0000509, -0.0090566, 0.0138727
4: -0.0038439, 0.0057439, -0.0011038, 0.0054707, -0.0093146, 0.0068476
5: 0.0097748, 0.0541063, 0.0097094, 0.0461924, -0.0364176, 0.0443968
6: -0.0080237, 0.0049228, -0.0062377, 0.0026866, -0.0107104, 0.0111605
7: -0.0181461, -0.0039147, -0.0155335, -0.0051321, -0.0130141, 0.0116189
8: -0.0083388, 0.0131751, -0.0070164, 0.0095369, -0.0178757, 0.0201915
9: -0.0000222, 0.0122006, 0.0010011, 0.0095247, -0.0095469, 0.0111994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0483332
time: 1.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0483332
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9389863, 0.9909836, 0.9389901, 0.9912318, -0.0522455, 0.0519935
1: -0.0113295, -0.0026218, -0.0113283, -0.0026218, -0.0087076, 0.0087066
2: 0.0083745, 0.0202836, 0.0082227, 0.0202828, -0.0119083, 0.0120609
3: -0.0090065, 0.0050606, -0.0090064, 0.0050585, -0.0140650, 0.0140671
4: -0.0038580, 0.0057453, -0.0038568, 0.0057452, -0.0096031, 0.0096020
5: 0.0097135, 0.0541469, 0.0093014, 0.0541435, -0.0444300, 0.0448456
6: -0.0080329, 0.0049342, -0.0080321, 0.0049333, -0.0129662, 0.0129663
7: -0.0181596, -0.0039084, -0.0181584, -0.0039089, -0.0142507, 0.0142500
8: -0.0083456, 0.0131939, -0.0083450, 0.0131922, -0.0215378, 0.0215389
9: -0.0000290, 0.0122143, -0.0000285, 0.0122131, -0.0122422, 0.0122428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0486003, upper bound: 0.0483244
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0486003, upper bound: 0.0483244
time: 1.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9275125, 0.9911223, 0.9479675, 0.9909773, -0.0634648, 0.0431548
1: -0.0146410, -0.0023774, -0.0087373, -0.0028131, -0.0118279, 0.0063600
2: 0.0067735, 0.0225398, 0.0085586, 0.0185175, -0.0117440, 0.0139249
3: -0.0092226, 0.0114749, -0.0088373, 0.0000399, -0.0092625, 0.0203122
4: -0.0073843, 0.0060968, -0.0010977, 0.0054701, -0.0128544, 0.0071945
5: 0.0094217, 0.0643316, 0.0097238, 0.0461749, -0.0367531, 0.0546077
6: -0.0103313, 0.0078120, -0.0062338, 0.0026817, -0.0130130, 0.0140458
7: -0.0215218, -0.0023416, -0.0155278, -0.0051348, -0.0163870, 0.0131861
8: -0.0100474, 0.0178760, -0.0070135, 0.0095289, -0.0195763, 0.0248895
9: -0.0017579, 0.0156579, 0.0010069, 0.0095188, -0.0112767, 0.0146510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0483921
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0483921
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9274671, 0.9911584, 0.9390104, 0.9912224, -0.0637553, 0.0521480
1: -0.0146540, -0.0023764, -0.0113225, -0.0026223, -0.0120318, 0.0089461
2: 0.0067672, 0.0225488, 0.0082349, 0.0202789, -0.0135117, 0.0143138
3: -0.0092235, 0.0115002, -0.0090060, 0.0050472, -0.0142707, 0.0205062
4: -0.0073982, 0.0060982, -0.0038506, 0.0057445, -0.0131428, 0.0099488
5: 0.0094124, 0.0643719, 0.0093168, 0.0541256, -0.0447132, 0.0550551
6: -0.0103404, 0.0078234, -0.0080281, 0.0049282, -0.0152686, 0.0158515
7: -0.0215351, -0.0023354, -0.0181525, -0.0039117, -0.0176234, 0.0158171
8: -0.0100541, 0.0178945, -0.0083420, 0.0131840, -0.0232382, 0.0262365
9: -0.0017647, 0.0156715, -0.0000254, 0.0122071, -0.0139718, 0.0156969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483244, upper bound: 0.0483822
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483244, upper bound: 0.0483822
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9482200, 0.9905944, 0.9192217, 0.9912378, -0.0430178, 0.0713727
1: -0.0086644, -0.0028184, -0.0170338, -0.0022008, -0.0064637, 0.0142154
2: 0.0090643, 0.0184678, 0.0056167, 0.0241702, -0.0151058, 0.0128511
3: -0.0088326, -0.0001013, -0.0093788, 0.0161097, -0.0249422, 0.0092775
4: -0.0010201, 0.0054624, -0.0099324, 0.0063508, -0.0073709, 0.0153948
5: 0.0103600, 0.0459506, 0.0077131, 0.0716910, -0.0613310, 0.0382375
6: -0.0061832, 0.0026183, -0.0119921, 0.0098914, -0.0160746, 0.0146104
7: -0.0154537, -0.0051693, -0.0239513, -0.0012095, -0.0142442, 0.0187821
8: -0.0069760, 0.0094258, -0.0112771, 0.0212593, -0.0282353, 0.0207029
9: 0.0012617, 0.0094430, -0.0030071, 0.0181462, -0.0168846, 0.0124502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0485183, upper bound: 0.0479187
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0485183, upper bound: 0.0479187
time: 1.56 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9392697, 0.9908380, 0.9191773, 0.9912749, -0.0520052, 0.0716607
1: -0.0112477, -0.0026278, -0.0170466, -0.0021998, -0.0090479, 0.0144188
2: 0.0084140, 0.0202279, 0.0056105, 0.0241789, -0.0157649, 0.0146174
3: -0.0090012, 0.0049022, -0.0093796, 0.0161345, -0.0251356, 0.0142819
4: -0.0037709, 0.0057366, -0.0099460, 0.0063522, -0.0101230, 0.0156826
5: 0.0099553, 0.0538954, 0.0077040, 0.0717303, -0.0617750, 0.0461915
6: -0.0079761, 0.0048632, -0.0120010, 0.0099026, -0.0178787, 0.0168642
7: -0.0180765, -0.0039471, -0.0239643, -0.0012034, -0.0168731, 0.0200173
8: -0.0083036, 0.0130782, -0.0112837, 0.0212774, -0.0295810, 0.0243619
9: 0.0000137, 0.0121293, -0.0030138, 0.0181595, -0.0181459, 0.0151431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0485023, upper bound: 0.0480361
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0485023, upper bound: 0.0480361
time: 1.56 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9366314, 0.9907673, 0.9192417, 0.9912260, -0.0545945, 0.0715256
1: -0.0120091, -0.0025716, -0.0170280, -0.0022012, -0.0098079, 0.0144564
2: 0.0080459, 0.0207467, 0.0056195, 0.0241663, -0.0161203, 0.0151271
3: -0.0090508, 0.0063771, -0.0093784, 0.0160985, -0.0251493, 0.0157555
4: -0.0045817, 0.0058174, -0.0099262, 0.0063502, -0.0109319, 0.0157436
5: 0.0100728, 0.0562372, 0.0077172, 0.0716732, -0.0616004, 0.0485200
6: -0.0085046, 0.0055249, -0.0119881, 0.0098864, -0.0183910, 0.0175130
7: -0.0188496, -0.0035868, -0.0239455, -0.0012122, -0.0176374, 0.0203586
8: -0.0086949, 0.0141548, -0.0112742, 0.0212511, -0.0299460, 0.0254289
9: -0.0003839, 0.0129211, -0.0030041, 0.0181402, -0.0185241, 0.0159252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482497, upper bound: 0.0479741
time: 1.69 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482497, upper bound: 0.0479741
time: 1.70 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9191970, 0.9912628, -0.0635228, 0.0718156
1: -0.0145753, -0.0023822, -0.0170409, -0.0022002, -0.0123751, 0.0146587
2: 0.0068053, 0.0224951, 0.0056133, 0.0241750, -0.0173697, 0.0168818
3: -0.0092183, 0.0113477, -0.0093793, 0.0161234, -0.0253417, 0.0207270
4: -0.0073144, 0.0060898, -0.0099399, 0.0063516, -0.0136659, 0.0160297
5: 0.0094686, 0.0641297, 0.0077081, 0.0717127, -0.0622441, 0.0564216
6: -0.0102857, 0.0077549, -0.0119970, 0.0098976, -0.0201833, 0.0197519
7: -0.0214552, -0.0023727, -0.0239585, -0.0012062, -0.0202490, 0.0215858
8: -0.0100137, 0.0177832, -0.0112808, 0.0212693, -0.0312830, 0.0290639
9: -0.0017236, 0.0155896, -0.0030108, 0.0181535, -0.0198772, 0.0186004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482373, upper bound: 0.0480973
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482373, upper bound: 0.0480973
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9192217, 0.9912378, 0.9482200, 0.9905944, -0.0713727, 0.0430178
1: -0.0170338, -0.0022008, -0.0086644, -0.0028184, -0.0142154, 0.0064637
2: 0.0056167, 0.0241702, 0.0090643, 0.0184678, -0.0128511, 0.0151058
3: -0.0093788, 0.0161097, -0.0088326, -0.0001013, -0.0092775, 0.0249422
4: -0.0099324, 0.0063508, -0.0010201, 0.0054624, -0.0153948, 0.0073709
5: 0.0077131, 0.0716910, 0.0103600, 0.0459506, -0.0382375, 0.0613310
6: -0.0119921, 0.0098914, -0.0061832, 0.0026183, -0.0146104, 0.0160746
7: -0.0239513, -0.0012095, -0.0154537, -0.0051693, -0.0187821, 0.0142442
8: -0.0112771, 0.0212593, -0.0069760, 0.0094258, -0.0207029, 0.0282353
9: -0.0030071, 0.0181462, 0.0012617, 0.0094430, -0.0124502, 0.0168846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0485183
time: 1.24 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0485207
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9191773, 0.9912749, 0.9392697, 0.9908380, -0.0716607, 0.0520052
1: -0.0170466, -0.0021998, -0.0112477, -0.0026278, -0.0144188, 0.0090479
2: 0.0056105, 0.0241789, 0.0084140, 0.0202279, -0.0146174, 0.0157649
3: -0.0093796, 0.0161345, -0.0090012, 0.0049022, -0.0142819, 0.0251356
4: -0.0099460, 0.0063522, -0.0037709, 0.0057366, -0.0156826, 0.0101230
5: 0.0077040, 0.0717303, 0.0099553, 0.0538954, -0.0461915, 0.0617750
6: -0.0120010, 0.0099026, -0.0079761, 0.0048632, -0.0168642, 0.0178787
7: -0.0239643, -0.0012034, -0.0180765, -0.0039471, -0.0200173, 0.0168731
8: -0.0112837, 0.0212774, -0.0083036, 0.0130782, -0.0243619, 0.0295810
9: -0.0030138, 0.0181595, 0.0000137, 0.0121293, -0.0151431, 0.0181459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480361, upper bound: 0.0485023
time: 1.43 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480361, upper bound: 0.0485050
time: 1.45 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9282309, 0.9908768, 0.9195111, 0.9908566, -0.0626257, 0.0713657
1: -0.0144336, -0.0023927, -0.0169503, -0.0022069, -0.0122267, 0.0145576
2: 0.0068738, 0.0223986, 0.0056571, 0.0241132, -0.0172395, 0.0167415
3: -0.0092091, 0.0110732, -0.0093733, 0.0159479, -0.0251570, 0.0204466
4: -0.0071635, 0.0060748, -0.0098434, 0.0063419, -0.0135054, 0.0159182
5: 0.0095698, 0.0636939, 0.0077727, 0.0714340, -0.0618642, 0.0559211
6: -0.0101874, 0.0076318, -0.0119341, 0.0098188, -0.0200062, 0.0195659
7: -0.0213113, -0.0024397, -0.0238665, -0.0012490, -0.0200623, 0.0214268
8: -0.0099408, 0.0175828, -0.0112342, 0.0211412, -0.0310820, 0.0288170
9: -0.0016496, 0.0154423, -0.0029635, 0.0180593, -0.0197090, 0.0184058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0482084
time: 1.54 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0482085
time: 1.62 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9194558, 0.9911248, 0.9194654, 0.9908925, -0.0714368, 0.0716594
1: -0.0169663, -0.0022057, -0.0169635, -0.0022060, -0.0147603, 0.0147577
2: 0.0056494, 0.0241242, 0.0056507, 0.0241222, -0.0184729, 0.0184734
3: -0.0093744, 0.0159789, -0.0093742, 0.0159734, -0.0253478, 0.0253531
4: -0.0098604, 0.0063436, -0.0098575, 0.0063433, -0.0162038, 0.0162011
5: 0.0077613, 0.0714832, 0.0077633, 0.0714747, -0.0637133, 0.0637199
6: -0.0119452, 0.0098327, -0.0119433, 0.0098303, -0.0217755, 0.0217760
7: -0.0238828, -0.0012415, -0.0238799, -0.0012428, -0.0226400, 0.0226385
8: -0.0112424, 0.0211638, -0.0112410, 0.0211599, -0.0324023, 0.0324048
9: -0.0029719, 0.0180760, -0.0029704, 0.0180731, -0.0210449, 0.0210464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480154, upper bound: 0.0483261
time: 1.45 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480154, upper bound: 0.0483270
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9192417, 0.9912260, 0.9366314, 0.9907673, -0.0715256, 0.0545945
1: -0.0170280, -0.0022012, -0.0120091, -0.0025716, -0.0144564, 0.0098079
2: 0.0056195, 0.0241663, 0.0080459, 0.0207467, -0.0151271, 0.0161203
3: -0.0093784, 0.0160985, -0.0090508, 0.0063771, -0.0157555, 0.0251493
4: -0.0099262, 0.0063502, -0.0045817, 0.0058174, -0.0157436, 0.0109319
5: 0.0077172, 0.0716732, 0.0100728, 0.0562372, -0.0485200, 0.0616004
6: -0.0119881, 0.0098864, -0.0085046, 0.0055249, -0.0175130, 0.0183910
7: -0.0239455, -0.0012122, -0.0188496, -0.0035868, -0.0203586, 0.0176374
8: -0.0112742, 0.0212511, -0.0086949, 0.0141548, -0.0254289, 0.0299460
9: -0.0030041, 0.0181402, -0.0003839, 0.0129211, -0.0159252, 0.0185241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0482497
time: 1.53 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0482896
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9191970, 0.9912628, 0.9277400, 0.9910126, -0.0718156, 0.0635228
1: -0.0170409, -0.0022002, -0.0145753, -0.0023822, -0.0146587, 0.0123751
2: 0.0056133, 0.0241750, 0.0068053, 0.0224951, -0.0168818, 0.0173697
3: -0.0093793, 0.0161234, -0.0092183, 0.0113477, -0.0207270, 0.0253417
4: -0.0099399, 0.0063516, -0.0073144, 0.0060898, -0.0160297, 0.0136659
5: 0.0077081, 0.0717127, 0.0094686, 0.0641297, -0.0564216, 0.0622441
6: -0.0119970, 0.0098976, -0.0102857, 0.0077549, -0.0197519, 0.0201833
7: -0.0239585, -0.0012062, -0.0214552, -0.0023727, -0.0215858, 0.0202490
8: -0.0112808, 0.0212693, -0.0100137, 0.0177832, -0.0290639, 0.0312830
9: -0.0030108, 0.0181535, -0.0017236, 0.0155896, -0.0186004, 0.0198772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480361, upper bound: 0.0482373
time: 1.56 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480361, upper bound: 0.0482756
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9282511, 0.9908645, 0.9087802, 0.9910029, -0.0627518, 0.0820843
1: -0.0144278, -0.0023931, -0.0200474, -0.0019784, -0.0124495, 0.0176543
2: 0.0068766, 0.0223946, 0.0041598, 0.0262234, -0.0193469, 0.0182348
3: -0.0092087, 0.0110620, -0.0095755, 0.0219468, -0.0311556, 0.0206375
4: -0.0071573, 0.0060742, -0.0131415, 0.0066707, -0.0138280, 0.0192156
5: 0.0095739, 0.0636761, 0.0055614, 0.0809594, -0.0713855, 0.0581147
6: -0.0101834, 0.0076268, -0.0140838, 0.0125103, -0.0226936, 0.0217105
7: -0.0213054, -0.0024425, -0.0270111, 0.0002163, -0.0215217, 0.0245686
8: -0.0099379, 0.0175746, -0.0128259, 0.0255203, -0.0354581, 0.0304005
9: -0.0016466, 0.0154362, -0.0045805, 0.0212800, -0.0229266, 0.0200167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0479326
time: 1.79 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0479684
time: 1.60 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9194765, 0.9911141, 0.9087361, 0.9910393, -0.0715628, 0.0823780
1: -0.0169602, -0.0022062, -0.0200601, -0.0019774, -0.0149828, 0.0178539
2: 0.0056523, 0.0241201, 0.0041537, 0.0262321, -0.0205798, 0.0199664
3: -0.0093740, 0.0159672, -0.0095763, 0.0219715, -0.0313455, 0.0255435
4: -0.0098540, 0.0063430, -0.0131550, 0.0066721, -0.0165261, 0.0194980
5: 0.0077656, 0.0714647, 0.0055523, 0.0809985, -0.0732329, 0.0659124
6: -0.0119411, 0.0098275, -0.0140926, 0.0125213, -0.0244624, 0.0239201
7: -0.0238766, -0.0012443, -0.0270240, 0.0002223, -0.0240989, 0.0257797
8: -0.0112393, 0.0211553, -0.0128324, 0.0255382, -0.0367775, 0.0339877
9: -0.0029687, 0.0180697, -0.0045871, 0.0212932, -0.0242619, 0.0226568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480155, upper bound: 0.0480429
time: 1.74 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0480155, upper bound: 0.0480843
time: 1.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.53 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0483332
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0483332
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0486003, upper bound: 0.0483244
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0486003, upper bound: 0.0483244
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0483921
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0483921
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0483244, upper bound: 0.0483822
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0483244, upper bound: 0.0483822
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0485183, upper bound: 0.0479187
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0485183, upper bound: 0.0479187
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0485023, upper bound: 0.0480361
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0485023, upper bound: 0.0480361
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0482497, upper bound: 0.0479741
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0482497, upper bound: 0.0479741
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0482373, upper bound: 0.0480973
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0482373, upper bound: 0.0480973
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0485183
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0485207
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480361, upper bound: 0.0485023
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480361, upper bound: 0.0485050
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0482084
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0482085
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480154, upper bound: 0.0483261
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480154, upper bound: 0.0483270
IS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0482497
IS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0482896
IS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480361, upper bound: 0.0482373
IS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480361, upper bound: 0.0482756
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0479326
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480343, upper bound: 0.0479684
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480155, upper bound: 0.0480429
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.53
Output dim: 0, lower bound: -0.0480155, upper bound: 0.0480843

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9390321, 0.9909468, 0.9482200, 0.9905944, -0.0515622, 0.0427268
1: -0.0113162, -0.0026227, -0.0086644, -0.0028184, -0.0084978, 0.0060417
2: 0.0083809, 0.0202746, 0.0090643, 0.0184678, -0.0100870, 0.0112103
3: -0.0090056, 0.0050350, -0.0088326, -0.0001013, -0.0089043, 0.0138676
4: -0.0038439, 0.0057439, -0.0010201, 0.0054624, -0.0093062, 0.0067639
5: 0.0097748, 0.0541063, 0.0103600, 0.0459506, -0.0361758, 0.0437463
6: -0.0080237, 0.0049228, -0.0061832, 0.0026183, -0.0106420, 0.0111060
7: -0.0181461, -0.0039147, -0.0154537, -0.0051693, -0.0129769, 0.0115391
8: -0.0083388, 0.0131751, -0.0069760, 0.0094258, -0.0177646, 0.0201511
9: -0.0000222, 0.0122006, 0.0012617, 0.0094430, -0.0094652, 0.0109389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447799, upper bound: 0.0445686
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9390321, 0.9909468, 0.9366314, 0.9907673, -0.0517352, 0.0543153
1: -0.0113162, -0.0026227, -0.0120091, -0.0025716, -0.0087446, 0.0093863
2: 0.0083809, 0.0202746, 0.0080459, 0.0207467, -0.0123658, 0.0122287
3: -0.0090056, 0.0050350, -0.0090508, 0.0063771, -0.0153827, 0.0140859
4: -0.0038439, 0.0057439, -0.0045817, 0.0058174, -0.0096613, 0.0103255
5: 0.0097748, 0.0541063, 0.0100728, 0.0562372, -0.0464624, 0.0440335
6: -0.0080237, 0.0049228, -0.0085046, 0.0055249, -0.0135486, 0.0134274
7: -0.0181461, -0.0039147, -0.0188496, -0.0035868, -0.0145593, 0.0149349
8: -0.0083388, 0.0131751, -0.0086949, 0.0141548, -0.0224936, 0.0218700
9: -0.0000222, 0.0122006, -0.0003839, 0.0129211, -0.0129432, 0.0125844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
time: 1.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447799, upper bound: 0.0445686
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9389863, 0.9909836, 0.9392697, 0.9908380, -0.0518517, 0.0517139
1: -0.0113295, -0.0026218, -0.0112477, -0.0026278, -0.0087017, 0.0086259
2: 0.0083745, 0.0202836, 0.0084140, 0.0202279, -0.0118534, 0.0118696
3: -0.0090065, 0.0050606, -0.0090012, 0.0049022, -0.0139087, 0.0140618
4: -0.0038580, 0.0057453, -0.0037709, 0.0057366, -0.0095945, 0.0095161
5: 0.0097135, 0.0541469, 0.0099553, 0.0538954, -0.0441819, 0.0441916
6: -0.0080329, 0.0049342, -0.0079761, 0.0048632, -0.0128961, 0.0129104
7: -0.0181596, -0.0039084, -0.0180765, -0.0039471, -0.0142125, 0.0141681
8: -0.0083456, 0.0131939, -0.0083036, 0.0130782, -0.0214238, 0.0214974
9: -0.0000290, 0.0122143, 0.0000137, 0.0121293, -0.0121583, 0.0122007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0482026
time: 1.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0483244
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9389863, 0.9909836, 0.9277400, 0.9910126, -0.0520263, 0.0632436
1: -0.0113295, -0.0026218, -0.0145753, -0.0023822, -0.0089472, 0.0119536
2: 0.0083745, 0.0202836, 0.0068053, 0.0224951, -0.0141206, 0.0134783
3: -0.0090065, 0.0050606, -0.0092183, 0.0113477, -0.0203542, 0.0142790
4: -0.0038580, 0.0057453, -0.0073144, 0.0060898, -0.0099478, 0.0130597
5: 0.0097135, 0.0541469, 0.0094686, 0.0641297, -0.0544162, 0.0446783
6: -0.0080329, 0.0049342, -0.0102857, 0.0077549, -0.0157878, 0.0152200
7: -0.0181596, -0.0039084, -0.0214552, -0.0023727, -0.0157869, 0.0175468
8: -0.0083456, 0.0131939, -0.0100137, 0.0177832, -0.0261287, 0.0232075
9: -0.0000290, 0.0122143, -0.0017236, 0.0155896, -0.0156187, 0.0139379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0482026
time: 1.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0483244
time: 1.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9275125, 0.9911223, 0.9482200, 0.9905944, -0.0630819, 0.0429023
1: -0.0146410, -0.0023774, -0.0086644, -0.0028184, -0.0118225, 0.0062871
2: 0.0067735, 0.0225398, 0.0090643, 0.0184678, -0.0116943, 0.0134755
3: -0.0092226, 0.0114749, -0.0088326, -0.0001013, -0.0091213, 0.0203074
4: -0.0073843, 0.0060968, -0.0010201, 0.0054624, -0.0128467, 0.0071169
5: 0.0094217, 0.0643316, 0.0103600, 0.0459506, -0.0365289, 0.0539716
6: -0.0103313, 0.0078120, -0.0061832, 0.0026183, -0.0129496, 0.0139952
7: -0.0215218, -0.0023416, -0.0154537, -0.0051693, -0.0163525, 0.0131121
8: -0.0100474, 0.0178760, -0.0069760, 0.0094258, -0.0194732, 0.0248520
9: -0.0017579, 0.0156579, 0.0012617, 0.0094430, -0.0112009, 0.0143962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0473347
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443862, upper bound: 0.0446128
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9275125, 0.9911223, 0.9366314, 0.9907673, -0.0632548, 0.0544909
1: -0.0146410, -0.0023774, -0.0120091, -0.0025716, -0.0120694, 0.0096317
2: 0.0067735, 0.0225398, 0.0080459, 0.0207467, -0.0139731, 0.0144939
3: -0.0092226, 0.0114749, -0.0090508, 0.0063771, -0.0155997, 0.0205257
4: -0.0073843, 0.0060968, -0.0045817, 0.0058174, -0.0132017, 0.0106785
5: 0.0094217, 0.0643316, 0.0100728, 0.0562372, -0.0468155, 0.0542588
6: -0.0103313, 0.0078120, -0.0085046, 0.0055249, -0.0158562, 0.0163166
7: -0.0215218, -0.0023416, -0.0188496, -0.0035868, -0.0179350, 0.0165080
8: -0.0100474, 0.0178760, -0.0086949, 0.0141548, -0.0242022, 0.0265709
9: -0.0017579, 0.0156579, -0.0003839, 0.0129211, -0.0146790, 0.0160418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446687
time: 1.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443862, upper bound: 0.0446128
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9274671, 0.9911584, 0.9392697, 0.9908380, -0.0633709, 0.0518888
1: -0.0146540, -0.0023764, -0.0112477, -0.0026278, -0.0120262, 0.0088713
2: 0.0067672, 0.0225488, 0.0084140, 0.0202279, -0.0134607, 0.0141348
3: -0.0092235, 0.0115002, -0.0090012, 0.0049022, -0.0141257, 0.0205013
4: -0.0073982, 0.0060982, -0.0037709, 0.0057366, -0.0131348, 0.0098690
5: 0.0094124, 0.0643719, 0.0099553, 0.0538954, -0.0444831, 0.0544165
6: -0.0103404, 0.0078234, -0.0079761, 0.0048632, -0.0152036, 0.0157995
7: -0.0215351, -0.0023354, -0.0180765, -0.0039471, -0.0175880, 0.0157411
8: -0.0100541, 0.0178945, -0.0083036, 0.0130782, -0.0231323, 0.0261981
9: -0.0017647, 0.0156715, 0.0000137, 0.0121293, -0.0138940, 0.0156579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0482560
time: 1.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0483822
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9274671, 0.9911584, 0.9277400, 0.9910126, -0.0635455, 0.0634184
1: -0.0146540, -0.0023764, -0.0145753, -0.0023822, -0.0122718, 0.0121989
2: 0.0067672, 0.0225488, 0.0068053, 0.0224951, -0.0157279, 0.0157435
3: -0.0092235, 0.0115002, -0.0092183, 0.0113477, -0.0205712, 0.0207185
4: -0.0073982, 0.0060982, -0.0073144, 0.0060898, -0.0134880, 0.0134126
5: 0.0094124, 0.0643719, 0.0094686, 0.0641297, -0.0547174, 0.0549033
6: -0.0103404, 0.0078234, -0.0102857, 0.0077549, -0.0180953, 0.0181091
7: -0.0215351, -0.0023354, -0.0214552, -0.0023727, -0.0191624, 0.0191197
8: -0.0100541, 0.0178945, -0.0100137, 0.0177832, -0.0278373, 0.0279082
9: -0.0017647, 0.0156715, -0.0017236, 0.0155896, -0.0173543, 0.0173951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0482560
time: 1.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0483822
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9482200, 0.9905944, 0.9195111, 0.9908566, -0.0426366, 0.0710832
1: -0.0086644, -0.0028184, -0.0169503, -0.0022069, -0.0064575, 0.0141318
2: 0.0090643, 0.0184678, 0.0056571, 0.0241132, -0.0150489, 0.0128107
3: -0.0088326, -0.0001013, -0.0093733, 0.0159479, -0.0247804, 0.0092720
4: -0.0010201, 0.0054624, -0.0098434, 0.0063419, -0.0073620, 0.0153058
5: 0.0103600, 0.0459506, 0.0077727, 0.0714340, -0.0610740, 0.0381779
6: -0.0061832, 0.0026183, -0.0119341, 0.0098188, -0.0160020, 0.0145524
7: -0.0154537, -0.0051693, -0.0238665, -0.0012490, -0.0142047, 0.0186972
8: -0.0069760, 0.0094258, -0.0112342, 0.0211412, -0.0281172, 0.0206600
9: 0.0012617, 0.0094430, -0.0029635, 0.0180593, -0.0167977, 0.0124065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0475262, upper bound: 0.0442552
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449235, upper bound: 0.0441284
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9482200, 0.9905944, 0.9087802, 0.9910029, -0.0427829, 0.0818142
1: -0.0086644, -0.0028184, -0.0200474, -0.0019784, -0.0066861, 0.0172290
2: 0.0090643, 0.0184678, 0.0041598, 0.0262234, -0.0171591, 0.0143080
3: -0.0088326, -0.0001013, -0.0095755, 0.0219468, -0.0307794, 0.0094742
4: -0.0010201, 0.0054624, -0.0131415, 0.0066707, -0.0076908, 0.0186038
5: 0.0103600, 0.0459506, 0.0055614, 0.0809594, -0.0705994, 0.0403893
6: -0.0061832, 0.0026183, -0.0140838, 0.0125103, -0.0186935, 0.0167021
7: -0.0154537, -0.0051693, -0.0270111, 0.0002163, -0.0156700, 0.0218418
8: -0.0069760, 0.0094258, -0.0128259, 0.0255203, -0.0324962, 0.0222517
9: 0.0012617, 0.0094430, -0.0045805, 0.0212800, -0.0200184, 0.0140235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0475262, upper bound: 0.0442552
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449235, upper bound: 0.0441284
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9392697, 0.9908380, 0.9194654, 0.9908925, -0.0516229, 0.0713726
1: -0.0112477, -0.0026278, -0.0169635, -0.0022060, -0.0090417, 0.0143357
2: 0.0084140, 0.0202279, 0.0056507, 0.0241222, -0.0157082, 0.0145772
3: -0.0090012, 0.0049022, -0.0093742, 0.0159734, -0.0249746, 0.0142765
4: -0.0037709, 0.0057366, -0.0098575, 0.0063433, -0.0101142, 0.0155941
5: 0.0099553, 0.0538954, 0.0077633, 0.0714747, -0.0615193, 0.0461321
6: -0.0079761, 0.0048632, -0.0119433, 0.0098303, -0.0178064, 0.0168065
7: -0.0180765, -0.0039471, -0.0238799, -0.0012428, -0.0168338, 0.0199329
8: -0.0083036, 0.0130782, -0.0112410, 0.0211599, -0.0294634, 0.0243192
9: 0.0000137, 0.0121293, -0.0029704, 0.0180731, -0.0180594, 0.0150997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483883, upper bound: 0.0480361
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483883, upper bound: 0.0480361
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9392697, 0.9908380, 0.9087361, 0.9910393, -0.0517696, 0.0821019
1: -0.0112477, -0.0026278, -0.0200601, -0.0019774, -0.0092703, 0.0174323
2: 0.0084140, 0.0202279, 0.0041537, 0.0262321, -0.0178181, 0.0160742
3: -0.0090012, 0.0049022, -0.0095763, 0.0219715, -0.0309726, 0.0144786
4: -0.0037709, 0.0057366, -0.0131550, 0.0066721, -0.0104429, 0.0188916
5: 0.0099553, 0.0538954, 0.0055523, 0.0809985, -0.0710432, 0.0483432
6: -0.0079761, 0.0048632, -0.0140926, 0.0125213, -0.0204974, 0.0189558
7: -0.0180765, -0.0039471, -0.0270240, 0.0002223, -0.0182989, 0.0230770
8: -0.0083036, 0.0130782, -0.0128324, 0.0255382, -0.0338418, 0.0259106
9: 0.0000137, 0.0121293, -0.0045871, 0.0212932, -0.0212796, 0.0167163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483883, upper bound: 0.0480361
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0483883, upper bound: 0.0480361
time: 1.60 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9366314, 0.9907673, 0.9195111, 0.9908566, -0.0542252, 0.0712562
1: -0.0120091, -0.0025716, -0.0169503, -0.0022069, -0.0098022, 0.0143787
2: 0.0080459, 0.0207467, 0.0056571, 0.0241132, -0.0160673, 0.0150896
3: -0.0090508, 0.0063771, -0.0093733, 0.0159479, -0.0249987, 0.0157504
4: -0.0045817, 0.0058174, -0.0098434, 0.0063419, -0.0109236, 0.0156608
5: 0.0100728, 0.0562372, 0.0077727, 0.0714340, -0.0613612, 0.0484645
6: -0.0085046, 0.0055249, -0.0119341, 0.0098188, -0.0183234, 0.0174590
7: -0.0188496, -0.0035868, -0.0238665, -0.0012490, -0.0176006, 0.0202797
8: -0.0086949, 0.0141548, -0.0112342, 0.0211412, -0.0298361, 0.0253890
9: -0.0003839, 0.0129211, -0.0029635, 0.0180593, -0.0184432, 0.0158846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0472107, upper bound: 0.0443075
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0444937, upper bound: 0.0441599
time: 1.55 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9366314, 0.9907673, 0.9087802, 0.9910029, -0.0543715, 0.0819871
1: -0.0120091, -0.0025716, -0.0200474, -0.0019784, -0.0100307, 0.0174758
2: 0.0080459, 0.0207467, 0.0041598, 0.0262234, -0.0181775, 0.0165869
3: -0.0090508, 0.0063771, -0.0095755, 0.0219468, -0.0309977, 0.0159526
4: -0.0045817, 0.0058174, -0.0131415, 0.0066707, -0.0112524, 0.0189589
5: 0.0100728, 0.0562372, 0.0055614, 0.0809594, -0.0708866, 0.0506759
6: -0.0085046, 0.0055249, -0.0140838, 0.0125103, -0.0210149, 0.0196086
7: -0.0188496, -0.0035868, -0.0270111, 0.0002163, -0.0190659, 0.0234243
8: -0.0086949, 0.0141548, -0.0128259, 0.0255203, -0.0342151, 0.0269807
9: -0.0003839, 0.0129211, -0.0045805, 0.0212800, -0.0216639, 0.0175015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0472107, upper bound: 0.0443075
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0444937, upper bound: 0.0441599
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9194654, 0.9908925, -0.0631526, 0.0715472
1: -0.0145753, -0.0023822, -0.0169635, -0.0022060, -0.0123694, 0.0145813
2: 0.0068053, 0.0224951, 0.0056507, 0.0241222, -0.0173170, 0.0168444
3: -0.0092183, 0.0113477, -0.0093742, 0.0159734, -0.0251918, 0.0207219
4: -0.0073144, 0.0060898, -0.0098575, 0.0063433, -0.0136577, 0.0159473
5: 0.0094686, 0.0641297, 0.0077633, 0.0714747, -0.0620060, 0.0563664
6: -0.0102857, 0.0077549, -0.0119433, 0.0098303, -0.0201160, 0.0196982
7: -0.0214552, -0.0023727, -0.0238799, -0.0012428, -0.0202124, 0.0215072
8: -0.0100137, 0.0177832, -0.0112410, 0.0211599, -0.0311735, 0.0290242
9: -0.0017236, 0.0155896, -0.0029704, 0.0180731, -0.0197967, 0.0185600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0481245, upper bound: 0.0480973
time: 1.54 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0481245, upper bound: 0.0480973
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9087361, 0.9910393, -0.0632993, 0.0822765
1: -0.0145753, -0.0023822, -0.0200601, -0.0019774, -0.0125979, 0.0176779
2: 0.0068053, 0.0224951, 0.0041537, 0.0262321, -0.0194268, 0.0183415
3: -0.0092183, 0.0113477, -0.0095763, 0.0219715, -0.0311898, 0.0209240
4: -0.0073144, 0.0060898, -0.0131550, 0.0066721, -0.0139864, 0.0192448
5: 0.0094686, 0.0641297, 0.0055523, 0.0809985, -0.0715299, 0.0585775
6: -0.0102857, 0.0077549, -0.0140926, 0.0125213, -0.0228071, 0.0218475
7: -0.0214552, -0.0023727, -0.0270240, 0.0002223, -0.0216775, 0.0246513
8: -0.0100137, 0.0177832, -0.0128324, 0.0255382, -0.0355519, 0.0306156
9: -0.0017236, 0.0155896, -0.0045871, 0.0212932, -0.0230169, 0.0201767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0481245, upper bound: 0.0480973
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0481245, upper bound: 0.0480973
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9195111, 0.9908566, 0.9482200, 0.9905944, -0.0710832, 0.0426366
1: -0.0169503, -0.0022069, -0.0086644, -0.0028184, -0.0141318, 0.0064575
2: 0.0056571, 0.0241132, 0.0090643, 0.0184678, -0.0128107, 0.0150489
3: -0.0093733, 0.0159479, -0.0088326, -0.0001013, -0.0092720, 0.0247804
4: -0.0098434, 0.0063419, -0.0010201, 0.0054624, -0.0153058, 0.0073620
5: 0.0077727, 0.0714340, 0.0103600, 0.0459506, -0.0381779, 0.0610740
6: -0.0119341, 0.0098188, -0.0061832, 0.0026183, -0.0145524, 0.0160020
7: -0.0238665, -0.0012490, -0.0154537, -0.0051693, -0.0186972, 0.0142047
8: -0.0112342, 0.0211412, -0.0069760, 0.0094258, -0.0206600, 0.0281172
9: -0.0029635, 0.0180593, 0.0012617, 0.0094430, -0.0124065, 0.0167977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
time: 1.44 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441284, upper bound: 0.0449235
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9087802, 0.9910029, 0.9482200, 0.9905944, -0.0818142, 0.0427829
1: -0.0200474, -0.0019784, -0.0086644, -0.0028184, -0.0172290, 0.0066861
2: 0.0041598, 0.0262234, 0.0090643, 0.0184678, -0.0143080, 0.0171591
3: -0.0095755, 0.0219468, -0.0088326, -0.0001013, -0.0094742, 0.0307794
4: -0.0131415, 0.0066707, -0.0010201, 0.0054624, -0.0186038, 0.0076908
5: 0.0055614, 0.0809594, 0.0103600, 0.0459506, -0.0403893, 0.0705994
6: -0.0140838, 0.0125103, -0.0061832, 0.0026183, -0.0167021, 0.0186935
7: -0.0270111, 0.0002163, -0.0154537, -0.0051693, -0.0218418, 0.0156700
8: -0.0128259, 0.0255203, -0.0069760, 0.0094258, -0.0222517, 0.0324962
9: -0.0045805, 0.0212800, 0.0012617, 0.0094430, -0.0140235, 0.0200184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
time: 1.29 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441284, upper bound: 0.0449235
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9194654, 0.9908925, 0.9392697, 0.9908380, -0.0713726, 0.0516229
1: -0.0169635, -0.0022060, -0.0112477, -0.0026278, -0.0143357, 0.0090417
2: 0.0056507, 0.0241222, 0.0084140, 0.0202279, -0.0145772, 0.0157082
3: -0.0093742, 0.0159734, -0.0090012, 0.0049022, -0.0142765, 0.0249746
4: -0.0098575, 0.0063433, -0.0037709, 0.0057366, -0.0155941, 0.0101142
5: 0.0077633, 0.0714747, 0.0099553, 0.0538954, -0.0461321, 0.0615193
6: -0.0119433, 0.0098303, -0.0079761, 0.0048632, -0.0168065, 0.0178064
7: -0.0238799, -0.0012428, -0.0180765, -0.0039471, -0.0199329, 0.0168338
8: -0.0112410, 0.0211599, -0.0083036, 0.0130782, -0.0243192, 0.0294634
9: -0.0029704, 0.0180731, 0.0000137, 0.0121293, -0.0150997, 0.0180594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0483883
time: 1.66 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0485023
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9087361, 0.9910393, 0.9392697, 0.9908380, -0.0821019, 0.0517696
1: -0.0200601, -0.0019774, -0.0112477, -0.0026278, -0.0174323, 0.0092703
2: 0.0041537, 0.0262321, 0.0084140, 0.0202279, -0.0160742, 0.0178181
3: -0.0095763, 0.0219715, -0.0090012, 0.0049022, -0.0144786, 0.0309726
4: -0.0131550, 0.0066721, -0.0037709, 0.0057366, -0.0188916, 0.0104429
5: 0.0055523, 0.0809985, 0.0099553, 0.0538954, -0.0483432, 0.0710432
6: -0.0140926, 0.0125213, -0.0079761, 0.0048632, -0.0189558, 0.0204974
7: -0.0270240, 0.0002223, -0.0180765, -0.0039471, -0.0230770, 0.0182989
8: -0.0128324, 0.0255382, -0.0083036, 0.0130782, -0.0259106, 0.0338418
9: -0.0045871, 0.0212932, 0.0000137, 0.0121293, -0.0167163, 0.0212796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0483898
time: 1.30 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0485050
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9285241, 0.9905008, 0.9195111, 0.9908566, -0.0623326, 0.0709897
1: -0.0143490, -0.0023989, -0.0169503, -0.0022069, -0.0121420, 0.0145513
2: 0.0069147, 0.0223409, 0.0056571, 0.0241132, -0.0171985, 0.0166838
3: -0.0092036, 0.0109093, -0.0093733, 0.0159479, -0.0251514, 0.0202827
4: -0.0070734, 0.0060658, -0.0098434, 0.0063419, -0.0134153, 0.0159092
5: 0.0096302, 0.0634336, 0.0077727, 0.0714340, -0.0618038, 0.0556609
6: -0.0101287, 0.0075583, -0.0119341, 0.0098188, -0.0199475, 0.0194924
7: -0.0212254, -0.0024798, -0.0238665, -0.0012490, -0.0199763, 0.0213867
8: -0.0098973, 0.0174632, -0.0112342, 0.0211412, -0.0310385, 0.0286974
9: -0.0016055, 0.0153543, -0.0029635, 0.0180593, -0.0196648, 0.0183178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0482084
time: 1.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0482084
time: 1.76 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9185466, 0.9906401, 0.9195111, 0.9908566, -0.0723100, 0.0711290
1: -0.0172287, -0.0021864, -0.0169503, -0.0022069, -0.0150217, 0.0147639
2: 0.0055225, 0.0243029, 0.0056571, 0.0241132, -0.0185907, 0.0186458
3: -0.0093915, 0.0164871, -0.0093733, 0.0159479, -0.0253394, 0.0258604
4: -0.0101399, 0.0063715, -0.0098434, 0.0063419, -0.0164818, 0.0162149
5: 0.0075740, 0.0722903, 0.0077727, 0.0714340, -0.0638600, 0.0645175
6: -0.0121274, 0.0100608, -0.0119341, 0.0098188, -0.0219462, 0.0219949
7: -0.0241492, -0.0011173, -0.0238665, -0.0012490, -0.0229002, 0.0227492
8: -0.0113773, 0.0215348, -0.0112342, 0.0211412, -0.0325185, 0.0327690
9: -0.0031089, 0.0183488, -0.0029635, 0.0180593, -0.0211682, 0.0213123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0482085
time: 1.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0482085
time: 1.61 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9197506, 0.9907448, 0.9194654, 0.9908925, -0.0711420, 0.0712794
1: -0.0168811, -0.0022120, -0.0169635, -0.0022060, -0.0146752, 0.0147514
2: 0.0056905, 0.0240662, 0.0056507, 0.0241222, -0.0184317, 0.0184154
3: -0.0093688, 0.0158140, -0.0093742, 0.0159734, -0.0253423, 0.0251882
4: -0.0097698, 0.0063346, -0.0098575, 0.0063433, -0.0161131, 0.0161921
5: 0.0078221, 0.0712213, 0.0077633, 0.0714747, -0.0636525, 0.0634580
6: -0.0118862, 0.0097587, -0.0119433, 0.0098303, -0.0217165, 0.0217020
7: -0.0237963, -0.0012817, -0.0238799, -0.0012428, -0.0225536, 0.0225982
8: -0.0111987, 0.0210434, -0.0112410, 0.0211599, -0.0323585, 0.0322844
9: -0.0029274, 0.0179875, -0.0029704, 0.0180731, -0.0210005, 0.0209579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483261
time: 1.64 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483261
time: 1.61 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9090129, 0.9908924, 0.9194654, 0.9908925, -0.0818797, 0.0714270
1: -0.0199802, -0.0019833, -0.0169635, -0.0022060, -0.0177743, 0.0149802
2: 0.0041923, 0.0261777, 0.0056507, 0.0241222, -0.0199300, 0.0205270
3: -0.0095711, 0.0218167, -0.0093742, 0.0159734, -0.0255445, 0.0311909
4: -0.0130699, 0.0066636, -0.0098575, 0.0063433, -0.0194133, 0.0165211
5: 0.0056093, 0.0807528, 0.0077633, 0.0714747, -0.0658653, 0.0729895
6: -0.0140371, 0.0124519, -0.0119433, 0.0098303, -0.0238674, 0.0243952
7: -0.0269429, 0.0001845, -0.0238799, -0.0012428, -0.0257001, 0.0240645
8: -0.0127913, 0.0254253, -0.0112410, 0.0211599, -0.0339512, 0.0366663
9: -0.0045454, 0.0212101, -0.0029704, 0.0180731, -0.0226185, 0.0241806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483270
time: 1.56 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483270
time: 1.84 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9195111, 0.9908566, 0.9366314, 0.9907673, -0.0712562, 0.0542252
1: -0.0169503, -0.0022069, -0.0120091, -0.0025716, -0.0143787, 0.0098022
2: 0.0056571, 0.0241132, 0.0080459, 0.0207467, -0.0150896, 0.0160673
3: -0.0093733, 0.0159479, -0.0090508, 0.0063771, -0.0157504, 0.0249987
4: -0.0098434, 0.0063419, -0.0045817, 0.0058174, -0.0156608, 0.0109236
5: 0.0077727, 0.0714340, 0.0100728, 0.0562372, -0.0484645, 0.0613612
6: -0.0119341, 0.0098188, -0.0085046, 0.0055249, -0.0174590, 0.0183234
7: -0.0238665, -0.0012490, -0.0188496, -0.0035868, -0.0202797, 0.0176006
8: -0.0112342, 0.0211412, -0.0086949, 0.0141548, -0.0253890, 0.0298361
9: -0.0029635, 0.0180593, -0.0003839, 0.0129211, -0.0158846, 0.0184432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
time: 1.48 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441280, upper bound: 0.0444937
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9087802, 0.9910029, 0.9366314, 0.9907673, -0.0819871, 0.0543715
1: -0.0200474, -0.0019784, -0.0120091, -0.0025716, -0.0174758, 0.0100307
2: 0.0041598, 0.0262234, 0.0080459, 0.0207467, -0.0165869, 0.0181775
3: -0.0095755, 0.0219468, -0.0090508, 0.0063771, -0.0159526, 0.0309977
4: -0.0131415, 0.0066707, -0.0045817, 0.0058174, -0.0189589, 0.0112524
5: 0.0055614, 0.0809594, 0.0100728, 0.0562372, -0.0506759, 0.0708866
6: -0.0140838, 0.0125103, -0.0085046, 0.0055249, -0.0196086, 0.0210149
7: -0.0270111, 0.0002163, -0.0188496, -0.0035868, -0.0234243, 0.0190659
8: -0.0128259, 0.0255203, -0.0086949, 0.0141548, -0.0269807, 0.0342151
9: -0.0045805, 0.0212800, -0.0003839, 0.0129211, -0.0175015, 0.0216639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
time: 1.09 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441280, upper bound: 0.0445252
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9194654, 0.9908925, 0.9277400, 0.9910126, -0.0715472, 0.0631526
1: -0.0169635, -0.0022060, -0.0145753, -0.0023822, -0.0145813, 0.0123694
2: 0.0056507, 0.0241222, 0.0068053, 0.0224951, -0.0168444, 0.0173170
3: -0.0093742, 0.0159734, -0.0092183, 0.0113477, -0.0207219, 0.0251918
4: -0.0098575, 0.0063433, -0.0073144, 0.0060898, -0.0159473, 0.0136577
5: 0.0077633, 0.0714747, 0.0094686, 0.0641297, -0.0563664, 0.0620060
6: -0.0119433, 0.0098303, -0.0102857, 0.0077549, -0.0196982, 0.0201160
7: -0.0238799, -0.0012428, -0.0214552, -0.0023727, -0.0215072, 0.0202124
8: -0.0112410, 0.0211599, -0.0100137, 0.0177832, -0.0290242, 0.0311735
9: -0.0029704, 0.0180731, -0.0017236, 0.0155896, -0.0185600, 0.0197967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0481245
time: 1.63 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0482373
time: 1.65 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9087361, 0.9910393, 0.9277400, 0.9910126, -0.0822765, 0.0632993
1: -0.0200601, -0.0019774, -0.0145753, -0.0023822, -0.0176779, 0.0125979
2: 0.0041537, 0.0262321, 0.0068053, 0.0224951, -0.0183415, 0.0194268
3: -0.0095763, 0.0219715, -0.0092183, 0.0113477, -0.0209240, 0.0311898
4: -0.0131550, 0.0066721, -0.0073144, 0.0060898, -0.0192448, 0.0139864
5: 0.0055523, 0.0809985, 0.0094686, 0.0641297, -0.0585775, 0.0715299
6: -0.0140926, 0.0125213, -0.0102857, 0.0077549, -0.0218475, 0.0228071
7: -0.0270240, 0.0002223, -0.0214552, -0.0023727, -0.0246513, 0.0216775
8: -0.0128324, 0.0255382, -0.0100137, 0.0177832, -0.0306156, 0.0355519
9: -0.0045871, 0.0212932, -0.0017236, 0.0155896, -0.0201767, 0.0230169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0481576
time: 1.96 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0482756
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9285241, 0.9905008, 0.9087802, 0.9910029, -0.0624788, 0.0817207
1: -0.0143490, -0.0023989, -0.0200474, -0.0019784, -0.0123706, 0.0176485
2: 0.0069147, 0.0223409, 0.0041598, 0.0262234, -0.0193087, 0.0181811
3: -0.0092036, 0.0109093, -0.0095755, 0.0219468, -0.0311504, 0.0204848
4: -0.0070734, 0.0060658, -0.0131415, 0.0066707, -0.0137441, 0.0192073
5: 0.0096302, 0.0634336, 0.0055614, 0.0809594, -0.0713292, 0.0578723
6: -0.0101287, 0.0075583, -0.0140838, 0.0125103, -0.0226390, 0.0216420
7: -0.0212254, -0.0024798, -0.0270111, 0.0002163, -0.0214417, 0.0245313
8: -0.0098973, 0.0174632, -0.0128259, 0.0255203, -0.0354176, 0.0302891
9: -0.0016055, 0.0153543, -0.0045805, 0.0212800, -0.0228855, 0.0199347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0479326
time: 1.63 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0479326
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9185466, 0.9906401, 0.9087802, 0.9910029, -0.0724563, 0.0818599
1: -0.0172287, -0.0021864, -0.0200474, -0.0019784, -0.0152503, 0.0178610
2: 0.0055225, 0.0243029, 0.0041598, 0.0262234, -0.0207009, 0.0201431
3: -0.0093915, 0.0164871, -0.0095755, 0.0219468, -0.0313384, 0.0260626
4: -0.0101399, 0.0063715, -0.0131415, 0.0066707, -0.0168106, 0.0195130
5: 0.0075740, 0.0722903, 0.0055614, 0.0809594, -0.0733854, 0.0667289
6: -0.0121274, 0.0100608, -0.0140838, 0.0125103, -0.0246376, 0.0241445
7: -0.0241492, -0.0011173, -0.0270111, 0.0002163, -0.0243655, 0.0258938
8: -0.0113773, 0.0215348, -0.0128259, 0.0255203, -0.0368975, 0.0343607
9: -0.0031089, 0.0183488, -0.0045805, 0.0212800, -0.0243889, 0.0229293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0479684
time: 1.65 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0479684
time: 1.63 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9197506, 0.9907448, 0.9087361, 0.9910393, -0.0712887, 0.0820087
1: -0.0168811, -0.0022120, -0.0200601, -0.0019774, -0.0149037, 0.0178481
2: 0.0056905, 0.0240662, 0.0041537, 0.0262321, -0.0205416, 0.0199125
3: -0.0093688, 0.0158140, -0.0095763, 0.0219715, -0.0313403, 0.0253903
4: -0.0097698, 0.0063346, -0.0131550, 0.0066721, -0.0164419, 0.0194896
5: 0.0078221, 0.0712213, 0.0055523, 0.0809985, -0.0731764, 0.0656691
6: -0.0118862, 0.0097587, -0.0140926, 0.0125213, -0.0244075, 0.0238513
7: -0.0237963, -0.0012817, -0.0270240, 0.0002223, -0.0240187, 0.0257423
8: -0.0111987, 0.0210434, -0.0128324, 0.0255382, -0.0367369, 0.0338758
9: -0.0029274, 0.0179875, -0.0045871, 0.0212932, -0.0242207, 0.0225745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0480429
time: 1.50 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0480429
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9090129, 0.9908924, 0.9087361, 0.9910393, -0.0820264, 0.0821564
1: -0.0199802, -0.0019833, -0.0200601, -0.0019774, -0.0180028, 0.0180768
2: 0.0041923, 0.0261777, 0.0041537, 0.0262321, -0.0220398, 0.0220240
3: -0.0095711, 0.0218167, -0.0095763, 0.0219715, -0.0315426, 0.0313930
4: -0.0130699, 0.0066636, -0.0131550, 0.0066721, -0.0197420, 0.0198186
5: 0.0056093, 0.0807528, 0.0055523, 0.0809985, -0.0753892, 0.0752006
6: -0.0140371, 0.0124519, -0.0140926, 0.0125213, -0.0265585, 0.0265445
7: -0.0269429, 0.0001845, -0.0270240, 0.0002223, -0.0271652, 0.0272086
8: -0.0127913, 0.0254253, -0.0128324, 0.0255382, -0.0383296, 0.0382577
9: -0.0045454, 0.0212101, -0.0045871, 0.0212932, -0.0258386, 0.0257972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0480843
time: 1.55 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0480843
time: 1.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.56 seconds
IS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
IS_A1_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0447799, upper bound: 0.0445686
IS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
IS_A1_B1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0447799, upper bound: 0.0445686
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0482026
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0483244
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0482026
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0484759, upper bound: 0.0483244
IS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0473347
IS_A1_B1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0443862, upper bound: 0.0446128
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446687
IS_A1_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0443862, upper bound: 0.0446128
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0482560
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0483822
IS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0482560
IS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0482026, upper bound: 0.0483822
IS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0475262, upper bound: 0.0442552
IS_A1_B2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0449235, upper bound: 0.0441284
IS_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0475262, upper bound: 0.0442552
IS_A1_B2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0449235, upper bound: 0.0441284
IS_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0483883, upper bound: 0.0480361
IS_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0483883, upper bound: 0.0480361
IS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0483883, upper bound: 0.0480361
IS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0483883, upper bound: 0.0480361
IS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0472107, upper bound: 0.0443075
IS_A1_B2_A2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0444937, upper bound: 0.0441599
IS_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0472107, upper bound: 0.0443075
IS_A1_B2_A2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0444937, upper bound: 0.0441599
IS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0481245, upper bound: 0.0480973
IS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0481245, upper bound: 0.0480973
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0481245, upper bound: 0.0480973
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0481245, upper bound: 0.0480973
IS_A2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
IS_A2_B1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0441284, upper bound: 0.0449235
IS_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
IS_A2_B1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0441284, upper bound: 0.0449235
IS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0483883
IS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0485023
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0483898
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0485050
IS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0482084
IS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0482084
IS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0482085
IS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0482085
IS_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483261
IS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483261
IS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483270
IS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0483270
IS_A2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
IS_A2_B2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0441280, upper bound: 0.0444937
IS_A2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
IS_A2_B2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0441280, upper bound: 0.0445252
IS_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0481245
IS_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0482373
IS_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0481576
IS_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479187, upper bound: 0.0482756
IS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0479326
IS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0479326
IS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0479684
IS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0479555, upper bound: 0.0479684
IS_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0480429
IS_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0480429
IS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0480843
IS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.56
Output dim: 0, lower bound: -0.0478965, upper bound: 0.0480843

## BFS IS instance: IS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9393641, 0.9907689, 0.9482200, 0.9905944, -0.0512303, 0.0425489
1: -0.0112204, -0.0026298, -0.0086644, -0.0028184, -0.0084019, 0.0060346
2: 0.0084272, 0.0202093, 0.0090643, 0.0184678, -0.0100406, 0.0111450
3: -0.0089994, 0.0048494, -0.0088326, -0.0001013, -0.0088981, 0.0136820
4: -0.0037418, 0.0057337, -0.0010201, 0.0054624, -0.0092042, 0.0067538
5: 0.0100701, 0.0538115, 0.0103600, 0.0459506, -0.0358805, 0.0434515
6: -0.0079572, 0.0048395, -0.0061832, 0.0026183, -0.0105755, 0.0110227
7: -0.0180488, -0.0039600, -0.0154537, -0.0051693, -0.0128796, 0.0114937
8: -0.0082895, 0.0130396, -0.0069760, 0.0094258, -0.0177153, 0.0200156
9: 0.0000279, 0.0121009, 0.0012617, 0.0094430, -0.0094151, 0.0108393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0449691
time: 1.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0450747
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9393641, 0.9907689, 0.9366314, 0.9907673, -0.0514032, 0.0541375
1: -0.0112204, -0.0026298, -0.0120091, -0.0025716, -0.0086488, 0.0093793
2: 0.0084272, 0.0202093, 0.0080459, 0.0207467, -0.0123195, 0.0121634
3: -0.0089994, 0.0048494, -0.0090508, 0.0063771, -0.0153764, 0.0139003
4: -0.0037418, 0.0057337, -0.0045817, 0.0058174, -0.0095592, 0.0103154
5: 0.0100701, 0.0538115, 0.0100728, 0.0562372, -0.0461671, 0.0437388
6: -0.0079572, 0.0048395, -0.0085046, 0.0055249, -0.0134821, 0.0133441
7: -0.0180488, -0.0039600, -0.0188496, -0.0035868, -0.0144620, 0.0148896
8: -0.0082895, 0.0130396, -0.0086949, 0.0141548, -0.0224443, 0.0217345
9: 0.0000279, 0.0121009, -0.0003839, 0.0129211, -0.0128932, 0.0124848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0445127
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9482200, 0.9905944, 0.9392697, 0.9908380, -0.0426180, 0.0513247
1: -0.0086644, -0.0028184, -0.0112477, -0.0026278, -0.0060366, 0.0084292
2: 0.0090643, 0.0184678, 0.0084140, 0.0202279, -0.0111636, 0.0100538
3: -0.0088326, -0.0001013, -0.0090012, 0.0049022, -0.0137348, 0.0088999
4: -0.0010201, 0.0054624, -0.0037709, 0.0057366, -0.0067567, 0.0092332
5: 0.0103600, 0.0459506, 0.0099553, 0.0538954, -0.0435354, 0.0359953
6: -0.0061832, 0.0026183, -0.0079761, 0.0048632, -0.0110464, 0.0105944
7: -0.0154537, -0.0051693, -0.0180765, -0.0039471, -0.0115067, 0.0129072
8: -0.0069760, 0.0094258, -0.0083036, 0.0130782, -0.0200542, 0.0177293
9: 0.0012617, 0.0094430, 0.0000137, 0.0121293, -0.0108676, 0.0094294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449691, upper bound: 0.0471856
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448429, upper bound: 0.0448429
time: 1.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9392697, 0.9908380, 0.9392697, 0.9908380, -0.0515683, 0.0515683
1: -0.0112477, -0.0026278, -0.0112477, -0.0026278, -0.0086199, 0.0086199
2: 0.0084140, 0.0202279, 0.0084140, 0.0202279, -0.0118139, 0.0118139
3: -0.0090012, 0.0049022, -0.0090012, 0.0049022, -0.0139034, 0.0139034
4: -0.0037709, 0.0057366, -0.0037709, 0.0057366, -0.0095074, 0.0095074
5: 0.0099553, 0.0538954, 0.0099553, 0.0538954, -0.0439401, 0.0439401
6: -0.0079761, 0.0048632, -0.0079761, 0.0048632, -0.0128393, 0.0128393
7: -0.0180765, -0.0039471, -0.0180765, -0.0039471, -0.0141295, 0.0141295
8: -0.0083036, 0.0130782, -0.0083036, 0.0130782, -0.0213818, 0.0213818
9: 0.0000137, 0.0121293, 0.0000137, 0.0121293, -0.0121156, 0.0121156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0450770
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448429, upper bound: 0.0450400
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9482200, 0.9905944, 0.9277400, 0.9910126, -0.0427926, 0.0628544
1: -0.0086644, -0.0028184, -0.0145753, -0.0023822, -0.0062822, 0.0117569
2: 0.0090643, 0.0184678, 0.0068053, 0.0224951, -0.0134308, 0.0116626
3: -0.0088326, -0.0001013, -0.0092183, 0.0113477, -0.0201803, 0.0091170
4: -0.0010201, 0.0054624, -0.0073144, 0.0060898, -0.0071099, 0.0127768
5: 0.0103600, 0.0459506, 0.0094686, 0.0641297, -0.0537697, 0.0364820
6: -0.0061832, 0.0026183, -0.0102857, 0.0077549, -0.0139381, 0.0129041
7: -0.0154537, -0.0051693, -0.0214552, -0.0023727, -0.0130810, 0.0162859
8: -0.0069760, 0.0094258, -0.0100137, 0.0177832, -0.0247591, 0.0194395
9: 0.0012617, 0.0094430, -0.0017236, 0.0155896, -0.0143280, 0.0111667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0445127
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447799, upper bound: 0.0443866
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9392697, 0.9908380, 0.9277400, 0.9910126, -0.0517429, 0.0630980
1: -0.0112477, -0.0026278, -0.0145753, -0.0023822, -0.0088655, 0.0119475
2: 0.0084140, 0.0202279, 0.0068053, 0.0224951, -0.0140811, 0.0134226
3: -0.0090012, 0.0049022, -0.0092183, 0.0113477, -0.0203489, 0.0141206
4: -0.0037709, 0.0057366, -0.0073144, 0.0060898, -0.0098607, 0.0130510
5: 0.0099553, 0.0538954, 0.0094686, 0.0641297, -0.0541744, 0.0444268
6: -0.0079761, 0.0048632, -0.0102857, 0.0077549, -0.0157311, 0.0151489
7: -0.0180765, -0.0039471, -0.0214552, -0.0023727, -0.0157038, 0.0175081
8: -0.0083036, 0.0130782, -0.0100137, 0.0177832, -0.0260867, 0.0230919
9: 0.0000137, 0.0121293, -0.0017236, 0.0155896, -0.0155760, 0.0138529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446194
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447799, upper bound: 0.0445754
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9275125, 0.9911223, 0.9485362, 0.9904261, -0.0629135, 0.0425861
1: -0.0146410, -0.0023774, -0.0085732, -0.0028252, -0.0118158, 0.0061958
2: 0.0067735, 0.0225398, 0.0092866, 0.0184057, -0.0116321, 0.0132533
3: -0.0092226, 0.0114749, -0.0088266, -0.0002781, -0.0089446, 0.0203015
4: -0.0073843, 0.0060968, -0.0009229, 0.0054527, -0.0128370, 0.0070197
5: 0.0094217, 0.0643316, 0.0106395, 0.0456699, -0.0362482, 0.0536921
6: -0.0103313, 0.0078120, -0.0061199, 0.0025390, -0.0128703, 0.0139319
7: -0.0215218, -0.0023416, -0.0153611, -0.0052125, -0.0163093, 0.0130194
8: -0.0100474, 0.0178760, -0.0069291, 0.0092967, -0.0193442, 0.0248051
9: -0.0017579, 0.0156579, 0.0013736, 0.0093481, -0.0111060, 0.0142843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470607
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0476103
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9278371, 0.9909477, 0.9366314, 0.9907673, -0.0629302, 0.0543162
1: -0.0145473, -0.0023843, -0.0120091, -0.0025716, -0.0119757, 0.0096248
2: 0.0068188, 0.0224760, 0.0080459, 0.0207467, -0.0139278, 0.0144301
3: -0.0092165, 0.0112934, -0.0090508, 0.0063771, -0.0155936, 0.0203442
4: -0.0072845, 0.0060869, -0.0045817, 0.0058174, -0.0131019, 0.0106685
5: 0.0094886, 0.0640435, 0.0100728, 0.0562372, -0.0467486, 0.0539708
6: -0.0102663, 0.0077306, -0.0085046, 0.0055249, -0.0157911, 0.0162352
7: -0.0214267, -0.0023859, -0.0188496, -0.0035868, -0.0178399, 0.0164637
8: -0.0099993, 0.0177436, -0.0086949, 0.0141548, -0.0241540, 0.0264384
9: -0.0017090, 0.0155605, -0.0003839, 0.0129211, -0.0146300, 0.0159443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0445656
time: 1.49 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446687
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9366314, 0.9907673, 0.9392697, 0.9908380, -0.0542066, 0.0514976
1: -0.0120091, -0.0025716, -0.0112477, -0.0026278, -0.0093813, 0.0086761
2: 0.0080459, 0.0207467, 0.0084140, 0.0202279, -0.0121820, 0.0123327
3: -0.0090508, 0.0063771, -0.0090012, 0.0049022, -0.0139531, 0.0153782
4: -0.0045817, 0.0058174, -0.0037709, 0.0057366, -0.0103183, 0.0095883
5: 0.0100728, 0.0562372, 0.0099553, 0.0538954, -0.0438227, 0.0462819
6: -0.0085046, 0.0055249, -0.0079761, 0.0048632, -0.0133678, 0.0135010
7: -0.0188496, -0.0035868, -0.0180765, -0.0039471, -0.0149025, 0.0144897
8: -0.0086949, 0.0141548, -0.0083036, 0.0130782, -0.0217731, 0.0224583
9: -0.0003839, 0.0129211, 0.0000137, 0.0121293, -0.0125131, 0.0129074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470071
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443866, upper bound: 0.0447799
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9392697, 0.9908380, -0.0630980, 0.0517429
1: -0.0145753, -0.0023822, -0.0112477, -0.0026278, -0.0119475, 0.0088655
2: 0.0068053, 0.0224951, 0.0084140, 0.0202279, -0.0134226, 0.0140811
3: -0.0092183, 0.0113477, -0.0090012, 0.0049022, -0.0141206, 0.0203489
4: -0.0073144, 0.0060898, -0.0037709, 0.0057366, -0.0130510, 0.0098607
5: 0.0094686, 0.0641297, 0.0099553, 0.0538954, -0.0444268, 0.0541744
6: -0.0102857, 0.0077549, -0.0079761, 0.0048632, -0.0151489, 0.0157311
7: -0.0214552, -0.0023727, -0.0180765, -0.0039471, -0.0175081, 0.0157038
8: -0.0100137, 0.0177832, -0.0083036, 0.0130782, -0.0230919, 0.0260867
9: -0.0017236, 0.0155896, 0.0000137, 0.0121293, -0.0138529, 0.0155760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0476103
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443866, upper bound: 0.0449944
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9366314, 0.9907673, 0.9277400, 0.9910126, -0.0543811, 0.0630273
1: -0.0120091, -0.0025716, -0.0145753, -0.0023822, -0.0096269, 0.0120037
2: 0.0080459, 0.0207467, 0.0068053, 0.0224951, -0.0144492, 0.0139414
3: -0.0090508, 0.0063771, -0.0092183, 0.0113477, -0.0203985, 0.0155954
4: -0.0045817, 0.0058174, -0.0073144, 0.0060898, -0.0106715, 0.0131318
5: 0.0100728, 0.0562372, 0.0094686, 0.0641297, -0.0540570, 0.0467686
6: -0.0085046, 0.0055249, -0.0102857, 0.0077549, -0.0162596, 0.0158106
7: -0.0188496, -0.0035868, -0.0214552, -0.0023727, -0.0164769, 0.0178683
8: -0.0086949, 0.0141548, -0.0100137, 0.0177832, -0.0264780, 0.0241685
9: -0.0003839, 0.0129211, -0.0017236, 0.0155896, -0.0159735, 0.0146447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0467338
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443862, upper bound: 0.0444178
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9277400, 0.9910126, -0.0632726, 0.0632726
1: -0.0145753, -0.0023822, -0.0145753, -0.0023822, -0.0121931, 0.0121931
2: 0.0068053, 0.0224951, 0.0068053, 0.0224951, -0.0156898, 0.0156898
3: -0.0092183, 0.0113477, -0.0092183, 0.0113477, -0.0205660, 0.0205660
4: -0.0073144, 0.0060898, -0.0073144, 0.0060898, -0.0134042, 0.0134042
5: 0.0094686, 0.0641297, 0.0094686, 0.0641297, -0.0546611, 0.0546611
6: -0.0102857, 0.0077549, -0.0102857, 0.0077549, -0.0180407, 0.0180407
7: -0.0214552, -0.0023727, -0.0214552, -0.0023727, -0.0190825, 0.0190825
8: -0.0100137, 0.0177832, -0.0100137, 0.0177832, -0.0277968, 0.0277968
9: -0.0017236, 0.0155896, -0.0017236, 0.0155896, -0.0173132, 0.0173132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446758
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443862, upper bound: 0.0444178
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9485362, 0.9904261, 0.9195111, 0.9908566, -0.0423204, 0.0709149
1: -0.0085732, -0.0028252, -0.0169503, -0.0022069, -0.0063663, 0.0141251
2: 0.0092866, 0.0184057, 0.0056571, 0.0241132, -0.0148267, 0.0127486
3: -0.0088266, -0.0002781, -0.0093733, 0.0159479, -0.0247745, 0.0090953
4: -0.0009229, 0.0054527, -0.0098434, 0.0063419, -0.0072648, 0.0152961
5: 0.0106395, 0.0456699, 0.0077727, 0.0714340, -0.0607945, 0.0378972
6: -0.0061199, 0.0025390, -0.0119341, 0.0098188, -0.0159387, 0.0144731
7: -0.0153611, -0.0052125, -0.0238665, -0.0012490, -0.0141120, 0.0186540
8: -0.0069291, 0.0092967, -0.0112342, 0.0211412, -0.0280703, 0.0205309
9: 0.0013736, 0.0093481, -0.0029635, 0.0180593, -0.0166858, 0.0123116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9485362, 0.9904261, 0.9087802, 0.9910029, -0.0424667, 0.0816459
1: -0.0085732, -0.0028252, -0.0200474, -0.0019784, -0.0065948, 0.0172222
2: 0.0092866, 0.0184057, 0.0041598, 0.0262234, -0.0169369, 0.0142459
3: -0.0088266, -0.0002781, -0.0095755, 0.0219468, -0.0307734, 0.0092974
4: -0.0009229, 0.0054527, -0.0131415, 0.0066707, -0.0075936, 0.0185941
5: 0.0106395, 0.0456699, 0.0055614, 0.0809594, -0.0703199, 0.0401086
6: -0.0061199, 0.0025390, -0.0140838, 0.0125103, -0.0186301, 0.0166228
7: -0.0153611, -0.0052125, -0.0270111, 0.0002163, -0.0155774, 0.0217986
8: -0.0069291, 0.0092967, -0.0128259, 0.0255203, -0.0324494, 0.0221226
9: 0.0013736, 0.0093481, -0.0045805, 0.0212800, -0.0199064, 0.0139286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
time: 1.60 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9392697, 0.9908380, 0.9285241, 0.9905008, -0.0512311, 0.0623139
1: -0.0112477, -0.0026278, -0.0143490, -0.0023989, -0.0088488, 0.0117212
2: 0.0084140, 0.0202279, 0.0069147, 0.0223409, -0.0139269, 0.0133132
3: -0.0090012, 0.0049022, -0.0092036, 0.0109093, -0.0199105, 0.0141058
4: -0.0037709, 0.0057366, -0.0070734, 0.0060658, -0.0098367, 0.0128099
5: 0.0099553, 0.0538954, 0.0096302, 0.0634336, -0.0534783, 0.0442653
6: -0.0079761, 0.0048632, -0.0101287, 0.0075583, -0.0155344, 0.0149919
7: -0.0180765, -0.0039471, -0.0212254, -0.0024798, -0.0155968, 0.0172783
8: -0.0083036, 0.0130782, -0.0098973, 0.0174632, -0.0257667, 0.0229756
9: 0.0000137, 0.0121293, -0.0016055, 0.0153543, -0.0153406, 0.0137347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471153, upper bound: 0.0449346
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448038, upper bound: 0.0448938
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9392697, 0.9908380, 0.9197506, 0.9907448, -0.0514751, 0.0710874
1: -0.0112477, -0.0026278, -0.0168811, -0.0022120, -0.0090356, 0.0142533
2: 0.0084140, 0.0202279, 0.0056905, 0.0240662, -0.0156522, 0.0145374
3: -0.0090012, 0.0049022, -0.0093688, 0.0158140, -0.0248151, 0.0142711
4: -0.0037709, 0.0057366, -0.0097698, 0.0063346, -0.0101055, 0.0155064
5: 0.0099553, 0.0538954, 0.0078221, 0.0712213, -0.0612660, 0.0460733
6: -0.0079761, 0.0048632, -0.0118862, 0.0097587, -0.0177349, 0.0167493
7: -0.0180765, -0.0039471, -0.0237963, -0.0012817, -0.0167948, 0.0198493
8: -0.0083036, 0.0130782, -0.0111987, 0.0210434, -0.0293470, 0.0242769
9: 0.0000137, 0.0121293, -0.0029274, 0.0179875, -0.0179738, 0.0150567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471153, upper bound: 0.0449352
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448038, upper bound: 0.0448950
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9392697, 0.9908380, 0.9185466, 0.9906401, -0.0513704, 0.0722914
1: -0.0112477, -0.0026278, -0.0172287, -0.0021864, -0.0090613, 0.0146009
2: 0.0084140, 0.0202279, 0.0055225, 0.0243029, -0.0158889, 0.0147054
3: -0.0090012, 0.0049022, -0.0093915, 0.0164871, -0.0254883, 0.0142938
4: -0.0037709, 0.0057366, -0.0101399, 0.0063715, -0.0101424, 0.0158764
5: 0.0099553, 0.0538954, 0.0075740, 0.0722903, -0.0623349, 0.0463214
6: -0.0079761, 0.0048632, -0.0121274, 0.0100608, -0.0180369, 0.0169905
7: -0.0180765, -0.0039471, -0.0241492, -0.0011173, -0.0169592, 0.0202021
8: -0.0083036, 0.0130782, -0.0113773, 0.0215348, -0.0298384, 0.0244555
9: 0.0000137, 0.0121293, -0.0031089, 0.0183488, -0.0183352, 0.0152381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469132, upper bound: 0.0443533
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447349, upper bound: 0.0443127
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9392697, 0.9908380, 0.9090129, 0.9908924, -0.0516227, 0.0818251
1: -0.0112477, -0.0026278, -0.0199802, -0.0019833, -0.0092644, 0.0173524
2: 0.0084140, 0.0202279, 0.0041923, 0.0261777, -0.0177637, 0.0160356
3: -0.0090012, 0.0049022, -0.0095711, 0.0218167, -0.0308179, 0.0144733
4: -0.0037709, 0.0057366, -0.0130699, 0.0066636, -0.0104344, 0.0188065
5: 0.0099553, 0.0538954, 0.0056093, 0.0807528, -0.0707975, 0.0482861
6: -0.0079761, 0.0048632, -0.0140371, 0.0124519, -0.0204280, 0.0189003
7: -0.0180765, -0.0039471, -0.0269429, 0.0001845, -0.0182611, 0.0229958
8: -0.0083036, 0.0130782, -0.0127913, 0.0254253, -0.0337288, 0.0258696
9: 0.0000137, 0.0121293, -0.0045454, 0.0212101, -0.0211965, 0.0166746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469132, upper bound: 0.0443594
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447349, upper bound: 0.0443172
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9369445, 0.9906062, 0.9195111, 0.9908566, -0.0539122, 0.0710951
1: -0.0119187, -0.0025783, -0.0169503, -0.0022069, -0.0097118, 0.0143720
2: 0.0080896, 0.0206851, 0.0056571, 0.0241132, -0.0160237, 0.0150280
3: -0.0090450, 0.0062021, -0.0093733, 0.0159479, -0.0249928, 0.0155754
4: -0.0044855, 0.0058078, -0.0098434, 0.0063419, -0.0108274, 0.0156512
5: 0.0103402, 0.0559594, 0.0077727, 0.0714340, -0.0610938, 0.0481866
6: -0.0084419, 0.0054464, -0.0119341, 0.0098188, -0.0182607, 0.0173805
7: -0.0187579, -0.0036296, -0.0238665, -0.0012490, -0.0175089, 0.0202369
8: -0.0086484, 0.0140271, -0.0112342, 0.0211412, -0.0297896, 0.0252613
9: -0.0003367, 0.0128271, -0.0029635, 0.0180593, -0.0183960, 0.0157906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9369445, 0.9906062, 0.9087802, 0.9910029, -0.0540584, 0.0818261
1: -0.0119187, -0.0025783, -0.0200474, -0.0019784, -0.0099404, 0.0174691
2: 0.0080896, 0.0206851, 0.0041598, 0.0262234, -0.0181339, 0.0165253
3: -0.0090450, 0.0062021, -0.0095755, 0.0219468, -0.0309918, 0.0157776
4: -0.0044855, 0.0058078, -0.0131415, 0.0066707, -0.0111562, 0.0189493
5: 0.0103402, 0.0559594, 0.0055614, 0.0809594, -0.0706192, 0.0503980
6: -0.0084419, 0.0054464, -0.0140838, 0.0125103, -0.0209522, 0.0195301
7: -0.0187579, -0.0036296, -0.0270111, 0.0002163, -0.0189742, 0.0233815
8: -0.0086484, 0.0140271, -0.0128259, 0.0255203, -0.0341687, 0.0268529
9: -0.0003367, 0.0128271, -0.0045805, 0.0212800, -0.0216167, 0.0174076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9285241, 0.9905008, -0.0627608, 0.0624885
1: -0.0145753, -0.0023822, -0.0143490, -0.0023989, -0.0121764, 0.0119668
2: 0.0068053, 0.0224951, 0.0069147, 0.0223409, -0.0155356, 0.0155804
3: -0.0092183, 0.0113477, -0.0092036, 0.0109093, -0.0201277, 0.0205513
4: -0.0073144, 0.0060898, -0.0070734, 0.0060658, -0.0133802, 0.0131632
5: 0.0094686, 0.0641297, 0.0096302, 0.0634336, -0.0539650, 0.0544995
6: -0.0102857, 0.0077549, -0.0101287, 0.0075583, -0.0178440, 0.0178836
7: -0.0214552, -0.0023727, -0.0212254, -0.0024798, -0.0189754, 0.0188527
8: -0.0100137, 0.0177832, -0.0098973, 0.0174632, -0.0274769, 0.0276805
9: -0.0017236, 0.0155896, -0.0016055, 0.0153543, -0.0170779, 0.0171951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466722, upper bound: 0.0449027
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443242, upper bound: 0.0448489
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9197506, 0.9907448, -0.0630048, 0.0712620
1: -0.0145753, -0.0023822, -0.0168811, -0.0022120, -0.0123633, 0.0144989
2: 0.0068053, 0.0224951, 0.0056905, 0.0240662, -0.0172609, 0.0168046
3: -0.0092183, 0.0113477, -0.0093688, 0.0158140, -0.0250323, 0.0207165
4: -0.0073144, 0.0060898, -0.0097698, 0.0063346, -0.0136490, 0.0158596
5: 0.0094686, 0.0641297, 0.0078221, 0.0712213, -0.0617527, 0.0563076
6: -0.0102857, 0.0077549, -0.0118862, 0.0097587, -0.0200445, 0.0196411
7: -0.0214552, -0.0023727, -0.0237963, -0.0012817, -0.0201734, 0.0214236
8: -0.0100137, 0.0177832, -0.0111987, 0.0210434, -0.0310571, 0.0289818
9: -0.0017236, 0.0155896, -0.0029274, 0.0179875, -0.0197111, 0.0185171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466722, upper bound: 0.0449037
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443242, upper bound: 0.0448501
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9185466, 0.9906401, -0.0629001, 0.0724660
1: -0.0145753, -0.0023822, -0.0172287, -0.0021864, -0.0123889, 0.0148465
2: 0.0068053, 0.0224951, 0.0055225, 0.0243029, -0.0174977, 0.0169726
3: -0.0092183, 0.0113477, -0.0093915, 0.0164871, -0.0257054, 0.0207392
4: -0.0073144, 0.0060898, -0.0101399, 0.0063715, -0.0136859, 0.0162297
5: 0.0094686, 0.0641297, 0.0075740, 0.0722903, -0.0628216, 0.0565557
6: -0.0102857, 0.0077549, -0.0121274, 0.0100608, -0.0203465, 0.0198823
7: -0.0214552, -0.0023727, -0.0241492, -0.0011173, -0.0203378, 0.0217765
8: -0.0100137, 0.0177832, -0.0113773, 0.0215348, -0.0315485, 0.0291604
9: -0.0017236, 0.0155896, -0.0031089, 0.0183488, -0.0200724, 0.0186985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466329, upper bound: 0.0444117
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443236, upper bound: 0.0443570
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9090129, 0.9908924, -0.0631524, 0.0819997
1: -0.0145753, -0.0023822, -0.0199802, -0.0019833, -0.0125920, 0.0175980
2: 0.0068053, 0.0224951, 0.0041923, 0.0261777, -0.0193724, 0.0183028
3: -0.0092183, 0.0113477, -0.0095711, 0.0218167, -0.0310351, 0.0209188
4: -0.0073144, 0.0060898, -0.0130699, 0.0066636, -0.0139780, 0.0191598
5: 0.0094686, 0.0641297, 0.0056093, 0.0807528, -0.0712842, 0.0585204
6: -0.0102857, 0.0077549, -0.0140371, 0.0124519, -0.0227377, 0.0217921
7: -0.0214552, -0.0023727, -0.0269429, 0.0001845, -0.0216397, 0.0245702
8: -0.0100137, 0.0177832, -0.0127913, 0.0254253, -0.0354390, 0.0305745
9: -0.0017236, 0.0155896, -0.0045454, 0.0212101, -0.0229338, 0.0201350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466329, upper bound: 0.0444164
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443236, upper bound: 0.0443585
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9195111, 0.9908566, 0.9485362, 0.9904261, -0.0709149, 0.0423204
1: -0.0169503, -0.0022069, -0.0085732, -0.0028252, -0.0141251, 0.0063663
2: 0.0056571, 0.0241132, 0.0092866, 0.0184057, -0.0127486, 0.0148267
3: -0.0093733, 0.0159479, -0.0088266, -0.0002781, -0.0090953, 0.0247745
4: -0.0098434, 0.0063419, -0.0009229, 0.0054527, -0.0152961, 0.0072648
5: 0.0077727, 0.0714340, 0.0106395, 0.0456699, -0.0378972, 0.0607945
6: -0.0119341, 0.0098188, -0.0061199, 0.0025390, -0.0144731, 0.0159387
7: -0.0238665, -0.0012490, -0.0153611, -0.0052125, -0.0186540, 0.0141120
8: -0.0112342, 0.0211412, -0.0069291, 0.0092967, -0.0205309, 0.0280703
9: -0.0029635, 0.0180593, 0.0013736, 0.0093481, -0.0123116, 0.0166858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0471528
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0476755
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9087802, 0.9910029, 0.9485362, 0.9904261, -0.0816459, 0.0424667
1: -0.0200474, -0.0019784, -0.0085732, -0.0028252, -0.0172222, 0.0065948
2: 0.0041598, 0.0262234, 0.0092866, 0.0184057, -0.0142459, 0.0169369
3: -0.0095755, 0.0219468, -0.0088266, -0.0002781, -0.0092974, 0.0307734
4: -0.0131415, 0.0066707, -0.0009229, 0.0054527, -0.0185941, 0.0075936
5: 0.0055614, 0.0809594, 0.0106395, 0.0456699, -0.0401086, 0.0703199
6: -0.0140838, 0.0125103, -0.0061199, 0.0025390, -0.0166228, 0.0186301
7: -0.0270111, 0.0002163, -0.0153611, -0.0052125, -0.0217986, 0.0155774
8: -0.0128259, 0.0255203, -0.0069291, 0.0092967, -0.0221226, 0.0324494
9: -0.0045805, 0.0212800, 0.0013736, 0.0093481, -0.0139286, 0.0199064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0469520
time: 1.49 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9285241, 0.9905008, 0.9392697, 0.9908380, -0.0623139, 0.0512311
1: -0.0143490, -0.0023989, -0.0112477, -0.0026278, -0.0117212, 0.0088488
2: 0.0069147, 0.0223409, 0.0084140, 0.0202279, -0.0133132, 0.0139269
3: -0.0092036, 0.0109093, -0.0090012, 0.0049022, -0.0141058, 0.0199105
4: -0.0070734, 0.0060658, -0.0037709, 0.0057366, -0.0128099, 0.0098367
5: 0.0096302, 0.0634336, 0.0099553, 0.0538954, -0.0442653, 0.0534783
6: -0.0101287, 0.0075583, -0.0079761, 0.0048632, -0.0149919, 0.0155344
7: -0.0212254, -0.0024798, -0.0180765, -0.0039471, -0.0172783, 0.0155968
8: -0.0098973, 0.0174632, -0.0083036, 0.0130782, -0.0229756, 0.0257667
9: -0.0016055, 0.0153543, 0.0000137, 0.0121293, -0.0137347, 0.0153406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0471153
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447068, upper bound: 0.0448038
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9197506, 0.9907448, 0.9392697, 0.9908380, -0.0710874, 0.0514751
1: -0.0168811, -0.0022120, -0.0112477, -0.0026278, -0.0142533, 0.0090356
2: 0.0056905, 0.0240662, 0.0084140, 0.0202279, -0.0145374, 0.0156522
3: -0.0093688, 0.0158140, -0.0090012, 0.0049022, -0.0142711, 0.0248151
4: -0.0097698, 0.0063346, -0.0037709, 0.0057366, -0.0155064, 0.0101055
5: 0.0078221, 0.0712213, 0.0099553, 0.0538954, -0.0460733, 0.0612660
6: -0.0118862, 0.0097587, -0.0079761, 0.0048632, -0.0167493, 0.0177349
7: -0.0237963, -0.0012817, -0.0180765, -0.0039471, -0.0198493, 0.0167948
8: -0.0111987, 0.0210434, -0.0083036, 0.0130782, -0.0242769, 0.0293470
9: -0.0029274, 0.0179875, 0.0000137, 0.0121293, -0.0150567, 0.0179738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0476755
time: 1.48 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447068, upper bound: 0.0448038
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9185466, 0.9906401, 0.9392697, 0.9908380, -0.0722914, 0.0513704
1: -0.0172287, -0.0021864, -0.0112477, -0.0026278, -0.0146009, 0.0090613
2: 0.0055225, 0.0243029, 0.0084140, 0.0202279, -0.0147054, 0.0158889
3: -0.0093915, 0.0164871, -0.0090012, 0.0049022, -0.0142938, 0.0254883
4: -0.0101399, 0.0063715, -0.0037709, 0.0057366, -0.0158764, 0.0101424
5: 0.0075740, 0.0722903, 0.0099553, 0.0538954, -0.0463214, 0.0623349
6: -0.0121274, 0.0100608, -0.0079761, 0.0048632, -0.0169905, 0.0180369
7: -0.0241492, -0.0011173, -0.0180765, -0.0039471, -0.0202021, 0.0169592
8: -0.0113773, 0.0215348, -0.0083036, 0.0130782, -0.0244555, 0.0298384
9: -0.0031089, 0.0183488, 0.0000137, 0.0121293, -0.0152381, 0.0183352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0469132
time: 1.54 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441284, upper bound: 0.0447349
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9090129, 0.9908924, 0.9392697, 0.9908380, -0.0818251, 0.0516227
1: -0.0199802, -0.0019833, -0.0112477, -0.0026278, -0.0173524, 0.0092644
2: 0.0041923, 0.0261777, 0.0084140, 0.0202279, -0.0160356, 0.0177637
3: -0.0095711, 0.0218167, -0.0090012, 0.0049022, -0.0144733, 0.0308179
4: -0.0130699, 0.0066636, -0.0037709, 0.0057366, -0.0188065, 0.0104344
5: 0.0056093, 0.0807528, 0.0099553, 0.0538954, -0.0482861, 0.0707975
6: -0.0140371, 0.0124519, -0.0079761, 0.0048632, -0.0189003, 0.0204280
7: -0.0269429, 0.0001845, -0.0180765, -0.0039471, -0.0229958, 0.0182611
8: -0.0127913, 0.0254253, -0.0083036, 0.0130782, -0.0258696, 0.0337288
9: -0.0045454, 0.0212101, 0.0000137, 0.0121293, -0.0166746, 0.0211965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441284, upper bound: 0.0449241
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9285241, 0.9905008, 0.9285241, 0.9905008, -0.0619767, 0.0619767
1: -0.0143490, -0.0023989, -0.0143490, -0.0023989, -0.0119501, 0.0119501
2: 0.0069147, 0.0223409, 0.0069147, 0.0223409, -0.0154262, 0.0154262
3: -0.0092036, 0.0109093, -0.0092036, 0.0109093, -0.0201129, 0.0201129
4: -0.0070734, 0.0060658, -0.0070734, 0.0060658, -0.0131392, 0.0131392
5: 0.0096302, 0.0634336, 0.0096302, 0.0634336, -0.0538035, 0.0538035
6: -0.0101287, 0.0075583, -0.0101287, 0.0075583, -0.0176869, 0.0176869
7: -0.0212254, -0.0024798, -0.0212254, -0.0024798, -0.0187456, 0.0187456
8: -0.0098973, 0.0174632, -0.0098973, 0.0174632, -0.0273605, 0.0273605
9: -0.0016055, 0.0153543, -0.0016055, 0.0153543, -0.0169597, 0.0169597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470147, upper bound: 0.0448382
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446937, upper bound: 0.0447248
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9285241, 0.9905008, 0.9197506, 0.9907448, -0.0622207, 0.0707502
1: -0.0143490, -0.0023989, -0.0168811, -0.0022120, -0.0121369, 0.0144822
2: 0.0069147, 0.0223409, 0.0056905, 0.0240662, -0.0171515, 0.0166504
3: -0.0092036, 0.0109093, -0.0093688, 0.0158140, -0.0250175, 0.0202781
4: -0.0070734, 0.0060658, -0.0097698, 0.0063346, -0.0134080, 0.0158356
5: 0.0096302, 0.0634336, 0.0078221, 0.0712213, -0.0615911, 0.0556115
6: -0.0101287, 0.0075583, -0.0118862, 0.0097587, -0.0198874, 0.0194444
7: -0.0212254, -0.0024798, -0.0237963, -0.0012817, -0.0199436, 0.0213166
8: -0.0098973, 0.0174632, -0.0111987, 0.0210434, -0.0309408, 0.0286619
9: -0.0016055, 0.0153543, -0.0029274, 0.0179875, -0.0195929, 0.0182817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0448068, upper bound: 0.0469904
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446937, upper bound: 0.0447248
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9185466, 0.9906401, 0.9285241, 0.9905008, -0.0719542, 0.0621160
1: -0.0172287, -0.0021864, -0.0143490, -0.0023989, -0.0148297, 0.0121626
2: 0.0055225, 0.0243029, 0.0069147, 0.0223409, -0.0168184, 0.0173882
3: -0.0093915, 0.0164871, -0.0092036, 0.0109093, -0.0203008, 0.0256907
4: -0.0101399, 0.0063715, -0.0070734, 0.0060658, -0.0162057, 0.0134449
5: 0.0075740, 0.0722903, 0.0096302, 0.0634336, -0.0558596, 0.0626601
6: -0.0121274, 0.0100608, -0.0101287, 0.0075583, -0.0196856, 0.0201894
7: -0.0241492, -0.0011173, -0.0212254, -0.0024798, -0.0216694, 0.0201080
8: -0.0113773, 0.0215348, -0.0098973, 0.0174632, -0.0288405, 0.0314321
9: -0.0031089, 0.0183488, -0.0016055, 0.0153543, -0.0184631, 0.0199543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0467822
time: 1.24 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0446489
time: 1.49 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9185466, 0.9906401, 0.9197506, 0.9907448, -0.0721982, 0.0708895
1: -0.0172287, -0.0021864, -0.0168811, -0.0022120, -0.0150166, 0.0146947
2: 0.0055225, 0.0243029, 0.0056905, 0.0240662, -0.0185437, 0.0186124
3: -0.0093915, 0.0164871, -0.0093688, 0.0158140, -0.0252055, 0.0258559
4: -0.0101399, 0.0063715, -0.0097698, 0.0063346, -0.0164745, 0.0161413
5: 0.0075740, 0.0722903, 0.0078221, 0.0712213, -0.0636473, 0.0644681
6: -0.0121274, 0.0100608, -0.0118862, 0.0097587, -0.0218861, 0.0219469
7: -0.0241492, -0.0011173, -0.0237963, -0.0012817, -0.0228674, 0.0226790
8: -0.0113773, 0.0215348, -0.0111987, 0.0210434, -0.0324207, 0.0327335
9: -0.0031089, 0.0183488, -0.0029274, 0.0179875, -0.0210963, 0.0212763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0467822
time: 1.23 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0446489
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9197506, 0.9907448, 0.9285241, 0.9905008, -0.0707502, 0.0622207
1: -0.0168811, -0.0022120, -0.0143490, -0.0023989, -0.0144822, 0.0121369
2: 0.0056905, 0.0240662, 0.0069147, 0.0223409, -0.0166504, 0.0171515
3: -0.0093688, 0.0158140, -0.0092036, 0.0109093, -0.0202781, 0.0250175
4: -0.0097698, 0.0063346, -0.0070734, 0.0060658, -0.0158356, 0.0134080
5: 0.0078221, 0.0712213, 0.0096302, 0.0634336, -0.0556115, 0.0615911
6: -0.0118862, 0.0097587, -0.0101287, 0.0075583, -0.0194444, 0.0198874
7: -0.0237963, -0.0012817, -0.0212254, -0.0024798, -0.0213166, 0.0199436
8: -0.0111987, 0.0210434, -0.0098973, 0.0174632, -0.0286619, 0.0309408
9: -0.0029274, 0.0179875, -0.0016055, 0.0153543, -0.0182817, 0.0195929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469761, upper bound: 0.0449315
time: 1.51 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446937, upper bound: 0.0449008
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9197506, 0.9907448, 0.9197506, 0.9907448, -0.0709942, 0.0709942
1: -0.0168811, -0.0022120, -0.0168811, -0.0022120, -0.0146691, 0.0146691
2: 0.0056905, 0.0240662, 0.0056905, 0.0240662, -0.0183756, 0.0183756
3: -0.0093688, 0.0158140, -0.0093688, 0.0158140, -0.0251828, 0.0251828
4: -0.0097698, 0.0063346, -0.0097698, 0.0063346, -0.0161044, 0.0161044
5: 0.0078221, 0.0712213, 0.0078221, 0.0712213, -0.0633992, 0.0633992
6: -0.0118862, 0.0097587, -0.0118862, 0.0097587, -0.0216449, 0.0216449
7: -0.0237963, -0.0012817, -0.0237963, -0.0012817, -0.0225146, 0.0225146
8: -0.0111987, 0.0210434, -0.0111987, 0.0210434, -0.0322421, 0.0322421
9: -0.0029274, 0.0179875, -0.0029274, 0.0179875, -0.0209149, 0.0209149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469761, upper bound: 0.0449317
time: 1.43 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446937, upper bound: 0.0449015
time: 1.49 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9090129, 0.9908924, 0.9285241, 0.9905008, -0.0814880, 0.0623683
1: -0.0199802, -0.0019833, -0.0143490, -0.0023989, -0.0175813, 0.0123657
2: 0.0041923, 0.0261777, 0.0069147, 0.0223409, -0.0181486, 0.0192630
3: -0.0095711, 0.0218167, -0.0092036, 0.0109093, -0.0204804, 0.0310203
4: -0.0130699, 0.0066636, -0.0070734, 0.0060658, -0.0191357, 0.0137370
5: 0.0056093, 0.0807528, 0.0096302, 0.0634336, -0.0578243, 0.0711226
6: -0.0140371, 0.0124519, -0.0101287, 0.0075583, -0.0215954, 0.0225806
7: -0.0269429, 0.0001845, -0.0212254, -0.0024798, -0.0244631, 0.0214099
8: -0.0127913, 0.0254253, -0.0098973, 0.0174632, -0.0302545, 0.0353226
9: -0.0045454, 0.0212101, -0.0016055, 0.0153543, -0.0198997, 0.0228156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0474100
time: 1.19 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0448383
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9090129, 0.9908924, 0.9197506, 0.9907448, -0.0817319, 0.0711418
1: -0.0199802, -0.0019833, -0.0168811, -0.0022120, -0.0177682, 0.0148978
2: 0.0041923, 0.0261777, 0.0056905, 0.0240662, -0.0198739, 0.0204871
3: -0.0095711, 0.0218167, -0.0093688, 0.0158140, -0.0253850, 0.0311855
4: -0.0130699, 0.0066636, -0.0097698, 0.0063346, -0.0194045, 0.0164334
5: 0.0056093, 0.0807528, 0.0078221, 0.0712213, -0.0656120, 0.0729307
6: -0.0140371, 0.0124519, -0.0118862, 0.0097587, -0.0237959, 0.0243381
7: -0.0269429, 0.0001845, -0.0237963, -0.0012817, -0.0256612, 0.0239809
8: -0.0127913, 0.0254253, -0.0111987, 0.0210434, -0.0338348, 0.0366240
9: -0.0045454, 0.0212101, -0.0029274, 0.0179875, -0.0225328, 0.0241376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0474100
time: 1.50 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0448389
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9195111, 0.9908566, 0.9369445, 0.9906062, -0.0710951, 0.0539122
1: -0.0169503, -0.0022069, -0.0119187, -0.0025783, -0.0143720, 0.0097118
2: 0.0056571, 0.0241132, 0.0080896, 0.0206851, -0.0150280, 0.0160237
3: -0.0093733, 0.0159479, -0.0090450, 0.0062021, -0.0155754, 0.0249928
4: -0.0098434, 0.0063419, -0.0044855, 0.0058078, -0.0156512, 0.0108274
5: 0.0077727, 0.0714340, 0.0103402, 0.0559594, -0.0481866, 0.0610938
6: -0.0119341, 0.0098188, -0.0084419, 0.0054464, -0.0173805, 0.0182607
7: -0.0238665, -0.0012490, -0.0187579, -0.0036296, -0.0202369, 0.0175089
8: -0.0112342, 0.0211412, -0.0086484, 0.0140271, -0.0252613, 0.0297896
9: -0.0029635, 0.0180593, -0.0003367, 0.0128271, -0.0157906, 0.0183960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0467036
time: 1.33 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0472362
time: 1.52 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9087802, 0.9910029, 0.9369445, 0.9906062, -0.0818261, 0.0540584
1: -0.0200474, -0.0019784, -0.0119187, -0.0025783, -0.0174691, 0.0099404
2: 0.0041598, 0.0262234, 0.0080896, 0.0206851, -0.0165253, 0.0181339
3: -0.0095755, 0.0219468, -0.0090450, 0.0062021, -0.0157776, 0.0309918
4: -0.0131415, 0.0066707, -0.0044855, 0.0058078, -0.0189493, 0.0111562
5: 0.0055614, 0.0809594, 0.0103402, 0.0559594, -0.0503980, 0.0706192
6: -0.0140838, 0.0125103, -0.0084419, 0.0054464, -0.0195301, 0.0209522
7: -0.0270111, 0.0002163, -0.0187579, -0.0036296, -0.0233815, 0.0189742
8: -0.0128259, 0.0255203, -0.0086484, 0.0140271, -0.0268529, 0.0341687
9: -0.0045805, 0.0212800, -0.0003367, 0.0128271, -0.0174076, 0.0216167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0466600
time: 1.22 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9285241, 0.9905008, 0.9277400, 0.9910126, -0.0624885, 0.0627608
1: -0.0143490, -0.0023989, -0.0145753, -0.0023822, -0.0119668, 0.0121764
2: 0.0069147, 0.0223409, 0.0068053, 0.0224951, -0.0155804, 0.0155356
3: -0.0092036, 0.0109093, -0.0092183, 0.0113477, -0.0205513, 0.0201277
4: -0.0070734, 0.0060658, -0.0073144, 0.0060898, -0.0131632, 0.0133802
5: 0.0096302, 0.0634336, 0.0094686, 0.0641297, -0.0544995, 0.0539650
6: -0.0101287, 0.0075583, -0.0102857, 0.0077549, -0.0178836, 0.0178440
7: -0.0212254, -0.0024798, -0.0214552, -0.0023727, -0.0188527, 0.0189754
8: -0.0098973, 0.0174632, -0.0100137, 0.0177832, -0.0276805, 0.0274769
9: -0.0016055, 0.0153543, -0.0017236, 0.0155896, -0.0171951, 0.0170779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0466722
time: 1.39 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446439, upper bound: 0.0443242
time: 1.52 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9197506, 0.9907448, 0.9277400, 0.9910126, -0.0712620, 0.0630048
1: -0.0168811, -0.0022120, -0.0145753, -0.0023822, -0.0144989, 0.0123633
2: 0.0056905, 0.0240662, 0.0068053, 0.0224951, -0.0168046, 0.0172609
3: -0.0093688, 0.0158140, -0.0092183, 0.0113477, -0.0207165, 0.0250323
4: -0.0097698, 0.0063346, -0.0073144, 0.0060898, -0.0158596, 0.0136490
5: 0.0078221, 0.0712213, 0.0094686, 0.0641297, -0.0563076, 0.0617527
6: -0.0118862, 0.0097587, -0.0102857, 0.0077549, -0.0196411, 0.0200445
7: -0.0237963, -0.0012817, -0.0214552, -0.0023727, -0.0214236, 0.0201734
8: -0.0111987, 0.0210434, -0.0100137, 0.0177832, -0.0289818, 0.0310571
9: -0.0029274, 0.0179875, -0.0017236, 0.0155896, -0.0185171, 0.0197111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0472362
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446439, upper bound: 0.0445011
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9185466, 0.9906401, 0.9277400, 0.9910126, -0.0724660, 0.0629001
1: -0.0172287, -0.0021864, -0.0145753, -0.0023822, -0.0148465, 0.0123889
2: 0.0055225, 0.0243029, 0.0068053, 0.0224951, -0.0169726, 0.0174977
3: -0.0093915, 0.0164871, -0.0092183, 0.0113477, -0.0207392, 0.0257054
4: -0.0101399, 0.0063715, -0.0073144, 0.0060898, -0.0162297, 0.0136859
5: 0.0075740, 0.0722903, 0.0094686, 0.0641297, -0.0565557, 0.0628216
6: -0.0121274, 0.0100608, -0.0102857, 0.0077549, -0.0198823, 0.0203465
7: -0.0241492, -0.0011173, -0.0214552, -0.0023727, -0.0217765, 0.0203378
8: -0.0113773, 0.0215348, -0.0100137, 0.0177832, -0.0291604, 0.0315485
9: -0.0031089, 0.0183488, -0.0017236, 0.0155896, -0.0186985, 0.0200724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0466329
time: 1.64 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441280, upper bound: 0.0443505
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9090129, 0.9908924, 0.9277400, 0.9910126, -0.0819997, 0.0631524
1: -0.0199802, -0.0019833, -0.0145753, -0.0023822, -0.0175980, 0.0125920
2: 0.0041923, 0.0261777, 0.0068053, 0.0224951, -0.0183028, 0.0193724
3: -0.0095711, 0.0218167, -0.0092183, 0.0113477, -0.0209188, 0.0310351
4: -0.0130699, 0.0066636, -0.0073144, 0.0060898, -0.0191598, 0.0139780
5: 0.0056093, 0.0807528, 0.0094686, 0.0641297, -0.0585204, 0.0712842
6: -0.0140371, 0.0124519, -0.0102857, 0.0077549, -0.0217921, 0.0227377
7: -0.0269429, 0.0001845, -0.0214552, -0.0023727, -0.0245702, 0.0216397
8: -0.0127913, 0.0254253, -0.0100137, 0.0177832, -0.0305745, 0.0354390
9: -0.0045454, 0.0212101, -0.0017236, 0.0155896, -0.0201350, 0.0229338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
time: 1.52 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441280, upper bound: 0.0445305
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9285241, 0.9905008, 0.9185466, 0.9906401, -0.0621160, 0.0719542
1: -0.0143490, -0.0023989, -0.0172287, -0.0021864, -0.0121626, 0.0148297
2: 0.0069147, 0.0223409, 0.0055225, 0.0243029, -0.0173882, 0.0168184
3: -0.0092036, 0.0109093, -0.0093915, 0.0164871, -0.0256907, 0.0203008
4: -0.0070734, 0.0060658, -0.0101399, 0.0063715, -0.0134449, 0.0162057
5: 0.0096302, 0.0634336, 0.0075740, 0.0722903, -0.0626601, 0.0558596
6: -0.0101287, 0.0075583, -0.0121274, 0.0100608, -0.0201894, 0.0196856
7: -0.0212254, -0.0024798, -0.0241492, -0.0011173, -0.0201080, 0.0216694
8: -0.0098973, 0.0174632, -0.0113773, 0.0215348, -0.0314321, 0.0288405
9: -0.0016055, 0.0153543, -0.0031089, 0.0183488, -0.0199543, 0.0184631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0468161, upper bound: 0.0442667
time: 1.26 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446283, upper bound: 0.0441575
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9285241, 0.9905008, 0.9090129, 0.9908924, -0.0623683, 0.0814880
1: -0.0143490, -0.0023989, -0.0199802, -0.0019833, -0.0123657, 0.0175813
2: 0.0069147, 0.0223409, 0.0041923, 0.0261777, -0.0192630, 0.0181486
3: -0.0092036, 0.0109093, -0.0095711, 0.0218167, -0.0310203, 0.0204804
4: -0.0070734, 0.0060658, -0.0130699, 0.0066636, -0.0137370, 0.0191357
5: 0.0096302, 0.0634336, 0.0056093, 0.0807528, -0.0711226, 0.0578243
6: -0.0101287, 0.0075583, -0.0140371, 0.0124519, -0.0225806, 0.0215954
7: -0.0212254, -0.0024798, -0.0269429, 0.0001845, -0.0214099, 0.0244631
8: -0.0098973, 0.0174632, -0.0127913, 0.0254253, -0.0353226, 0.0302545
9: -0.0016055, 0.0153543, -0.0045454, 0.0212101, -0.0228156, 0.0198997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0468161, upper bound: 0.0442667
time: 1.23 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446283, upper bound: 0.0441575
time: 1.57 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9185466, 0.9906401, 0.9185466, 0.9906401, -0.0720935, 0.0720935
1: -0.0172287, -0.0021864, -0.0172287, -0.0021864, -0.0150423, 0.0150423
2: 0.0055225, 0.0243029, 0.0055225, 0.0243029, -0.0187804, 0.0187804
3: -0.0093915, 0.0164871, -0.0093915, 0.0164871, -0.0258786, 0.0258786
4: -0.0101399, 0.0063715, -0.0101399, 0.0063715, -0.0165114, 0.0165114
5: 0.0075740, 0.0722903, 0.0075740, 0.0722903, -0.0647162, 0.0647162
6: -0.0121274, 0.0100608, -0.0121274, 0.0100608, -0.0221881, 0.0221881
7: -0.0241492, -0.0011173, -0.0241492, -0.0011173, -0.0230319, 0.0230319
8: -0.0113773, 0.0215348, -0.0113773, 0.0215348, -0.0329121, 0.0329121
9: -0.0031089, 0.0183488, -0.0031089, 0.0183488, -0.0214577, 0.0214577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0464751, upper bound: 0.0443010
time: 1.25 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0441822
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9185466, 0.9906401, 0.9090129, 0.9908924, -0.0723458, 0.0816272
1: -0.0172287, -0.0021864, -0.0199802, -0.0019833, -0.0152454, 0.0177938
2: 0.0055225, 0.0243029, 0.0041923, 0.0261777, -0.0206552, 0.0201107
3: -0.0093915, 0.0164871, -0.0095711, 0.0218167, -0.0312082, 0.0260582
4: -0.0101399, 0.0063715, -0.0130699, 0.0066636, -0.0168035, 0.0194414
5: 0.0075740, 0.0722903, 0.0056093, 0.0807528, -0.0731788, 0.0666810
6: -0.0121274, 0.0100608, -0.0140371, 0.0124519, -0.0245793, 0.0240979
7: -0.0241492, -0.0011173, -0.0269429, 0.0001845, -0.0243337, 0.0258256
8: -0.0113773, 0.0215348, -0.0127913, 0.0254253, -0.0368026, 0.0343262
9: -0.0031089, 0.0183488, -0.0045454, 0.0212101, -0.0243190, 0.0228942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0464602
time: 1.49 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0441822
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9197506, 0.9907448, 0.9185466, 0.9906401, -0.0708895, 0.0721982
1: -0.0168811, -0.0022120, -0.0172287, -0.0021864, -0.0146947, 0.0150166
2: 0.0056905, 0.0240662, 0.0055225, 0.0243029, -0.0186124, 0.0185437
3: -0.0093688, 0.0158140, -0.0093915, 0.0164871, -0.0258559, 0.0252055
4: -0.0097698, 0.0063346, -0.0101399, 0.0063715, -0.0161413, 0.0164745
5: 0.0078221, 0.0712213, 0.0075740, 0.0722903, -0.0644681, 0.0636473
6: -0.0118862, 0.0097587, -0.0121274, 0.0100608, -0.0219469, 0.0218861
7: -0.0237963, -0.0012817, -0.0241492, -0.0011173, -0.0226790, 0.0228674
8: -0.0111987, 0.0210434, -0.0113773, 0.0215348, -0.0327335, 0.0324207
9: -0.0029274, 0.0179875, -0.0031089, 0.0183488, -0.0212763, 0.0210963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467791, upper bound: 0.0443563
time: 1.50 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446283, upper bound: 0.0443263
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9197506, 0.9907448, 0.9090129, 0.9908924, -0.0711418, 0.0817319
1: -0.0168811, -0.0022120, -0.0199802, -0.0019833, -0.0148978, 0.0177682
2: 0.0056905, 0.0240662, 0.0041923, 0.0261777, -0.0204871, 0.0198739
3: -0.0093688, 0.0158140, -0.0095711, 0.0218167, -0.0311855, 0.0253850
4: -0.0097698, 0.0063346, -0.0130699, 0.0066636, -0.0164334, 0.0194045
5: 0.0078221, 0.0712213, 0.0056093, 0.0807528, -0.0729307, 0.0656120
6: -0.0118862, 0.0097587, -0.0140371, 0.0124519, -0.0243381, 0.0237959
7: -0.0237963, -0.0012817, -0.0269429, 0.0001845, -0.0239809, 0.0256612
8: -0.0111987, 0.0210434, -0.0127913, 0.0254253, -0.0366240, 0.0338348
9: -0.0029274, 0.0179875, -0.0045454, 0.0212101, -0.0241376, 0.0225328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467791, upper bound: 0.0443593
time: 1.43 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0446283, upper bound: 0.0443280
time: 1.63 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9090129, 0.9908924, 0.9185466, 0.9906401, -0.0816272, 0.0723458
1: -0.0199802, -0.0019833, -0.0172287, -0.0021864, -0.0177938, 0.0152454
2: 0.0041923, 0.0261777, 0.0055225, 0.0243029, -0.0201107, 0.0206552
3: -0.0095711, 0.0218167, -0.0093915, 0.0164871, -0.0260582, 0.0312082
4: -0.0130699, 0.0066636, -0.0101399, 0.0063715, -0.0194414, 0.0168035
5: 0.0056093, 0.0807528, 0.0075740, 0.0722903, -0.0666810, 0.0731788
6: -0.0140371, 0.0124519, -0.0121274, 0.0100608, -0.0240979, 0.0245793
7: -0.0269429, 0.0001845, -0.0241492, -0.0011173, -0.0258256, 0.0243337
8: -0.0127913, 0.0254253, -0.0113773, 0.0215348, -0.0343262, 0.0368026
9: -0.0045454, 0.0212101, -0.0031089, 0.0183488, -0.0228942, 0.0243190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0464524, upper bound: 0.0443987
time: 1.41 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0443529
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9090129, 0.9908924, 0.9090129, 0.9908924, -0.0818796, 0.0818796
1: -0.0199802, -0.0019833, -0.0199802, -0.0019833, -0.0179969, 0.0179969
2: 0.0041923, 0.0261777, 0.0041923, 0.0261777, -0.0219854, 0.0219854
3: -0.0095711, 0.0218167, -0.0095711, 0.0218167, -0.0313878, 0.0313878
4: -0.0130699, 0.0066636, -0.0130699, 0.0066636, -0.0197335, 0.0197335
5: 0.0056093, 0.0807528, 0.0056093, 0.0807528, -0.0751435, 0.0751435
6: -0.0140371, 0.0124519, -0.0140371, 0.0124519, -0.0264890, 0.0264890
7: -0.0269429, 0.0001845, -0.0269429, 0.0001845, -0.0271274, 0.0271274
8: -0.0127913, 0.0254253, -0.0127913, 0.0254253, -0.0382166, 0.0382166
9: -0.0045454, 0.0212101, -0.0045454, 0.0212101, -0.0257555, 0.0257555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0464524, upper bound: 0.0444019
time: 1.48 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0443557
time: 1.49 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.41 seconds
IS_A1_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0449691
IS_A1_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0450747
IS_A1_B1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0445127
IS_A1_B1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
IS_A1_B1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0449691, upper bound: 0.0471856
IS_A1_B1_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0448429, upper bound: 0.0448429
IS_A1_B1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0450770
IS_A1_B1_A1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0448429, upper bound: 0.0450400
IS_A1_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0445127
IS_A1_B1_A1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447799, upper bound: 0.0443866
IS_A1_B1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446194
IS_A1_B1_A1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447799, upper bound: 0.0445754
IS_A1_B1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470607
IS_A1_B1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0476103
IS_A1_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0445656
IS_A1_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446687
IS_A1_B1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470071
IS_A1_B1_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0443866, upper bound: 0.0447799
IS_A1_B1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0476103
IS_A1_B1_A2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0443866, upper bound: 0.0449944
IS_A1_B1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0467338
IS_A1_B1_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0443862, upper bound: 0.0444178
IS_A1_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446758
IS_A1_B1_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0443862, upper bound: 0.0444178
IS_A1_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
IS_A1_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
IS_A1_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
IS_A1_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
IS_A1_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0471153, upper bound: 0.0449346
IS_A1_B2_A1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0448038, upper bound: 0.0448938
IS_A1_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0471153, upper bound: 0.0449352
IS_A1_B2_A1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0448038, upper bound: 0.0448950
IS_A1_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0469132, upper bound: 0.0443533
IS_A1_B2_A1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447349, upper bound: 0.0443127
IS_A1_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0469132, upper bound: 0.0443594
IS_A1_B2_A1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447349, upper bound: 0.0443172
IS_A1_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
IS_A1_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
IS_A1_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
IS_A1_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
IS_A1_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0466722, upper bound: 0.0449027
IS_A1_B2_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0443242, upper bound: 0.0448489
IS_A1_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0466722, upper bound: 0.0449037
IS_A1_B2_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0443242, upper bound: 0.0448501
IS_A1_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0466329, upper bound: 0.0444117
IS_A1_B2_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0443236, upper bound: 0.0443570
IS_A1_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0466329, upper bound: 0.0444164
IS_A1_B2_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0443236, upper bound: 0.0443585
IS_A2_B1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0471528
IS_A2_B1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0476755
IS_A2_B1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0469520
IS_A2_B1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
IS_A2_B1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0471153
IS_A2_B1_B1_B2_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447068, upper bound: 0.0448038
IS_A2_B1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0476755
IS_A2_B1_B1_B2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447068, upper bound: 0.0448038
IS_A2_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0469132
IS_A2_B1_B1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441284, upper bound: 0.0447349
IS_A2_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
IS_A2_B1_B1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441284, upper bound: 0.0449241
IS_A2_B1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0470147, upper bound: 0.0448382
IS_A2_B1_B2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446937, upper bound: 0.0447248
IS_A2_B1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0448068, upper bound: 0.0469904
IS_A2_B1_B2_A1_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446937, upper bound: 0.0447248
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0467822
IS_A2_B1_B2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0446489
IS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0467822
IS_A2_B1_B2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0446489
IS_A2_B1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0469761, upper bound: 0.0449315
IS_A2_B1_B2_A2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446937, upper bound: 0.0449008
IS_A2_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0469761, upper bound: 0.0449317
IS_A2_B1_B2_A2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446937, upper bound: 0.0449015
IS_A2_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0474100
IS_A2_B1_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0448383
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0474100
IS_A2_B1_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0448389
IS_A2_B2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0467036
IS_A2_B2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0472362
IS_A2_B2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0466600
IS_A2_B2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
IS_A2_B2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0466722
IS_A2_B2_B1_B2_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446439, upper bound: 0.0443242
IS_A2_B2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0472362
IS_A2_B2_B1_B2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446439, upper bound: 0.0445011
IS_A2_B2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0466329
IS_A2_B2_B1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441280, upper bound: 0.0443505
IS_A2_B2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
IS_A2_B2_B1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441280, upper bound: 0.0445305
IS_A2_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0468161, upper bound: 0.0442667
IS_A2_B2_B2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446283, upper bound: 0.0441575
IS_A2_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0468161, upper bound: 0.0442667
IS_A2_B2_B2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446283, upper bound: 0.0441575
IS_A2_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0464751, upper bound: 0.0443010
IS_A2_B2_B2_A1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0441822
IS_A2_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0464602
IS_A2_B2_B2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0441822
IS_A2_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0467791, upper bound: 0.0443563
IS_A2_B2_B2_A2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446283, upper bound: 0.0443263
IS_A2_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0467791, upper bound: 0.0443593
IS_A2_B2_B2_A2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0446283, upper bound: 0.0443280
IS_A2_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0464524, upper bound: 0.0443987
IS_A2_B2_B2_A2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0443529
IS_A2_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0464524, upper bound: 0.0444019
IS_A2_B2_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 0, lower bound: -0.0441185, upper bound: 0.0443557

## BFS IS instance: IS_A1_B1_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9485362, 0.9904261, 0.9482200, 0.9905944, -0.0420582, 0.0422060
1: -0.0085732, -0.0028252, -0.0086644, -0.0028184, -0.0057548, 0.0058393
2: 0.0092866, 0.0184057, 0.0090643, 0.0184678, -0.0091813, 0.0093413
3: -0.0088266, -0.0002781, -0.0088326, -0.0001013, -0.0087253, 0.0085545
4: -0.0009229, 0.0054527, -0.0010201, 0.0054624, -0.0063853, 0.0064728
5: 0.0106395, 0.0456699, 0.0103600, 0.0459506, -0.0353111, 0.0353100
6: -0.0061199, 0.0025390, -0.0061832, 0.0026183, -0.0087382, 0.0087222
7: -0.0153611, -0.0052125, -0.0154537, -0.0051693, -0.0101918, 0.0102413
8: -0.0069291, 0.0092967, -0.0069760, 0.0094258, -0.0163549, 0.0162727
9: 0.0013736, 0.0093481, 0.0012617, 0.0094430, -0.0080695, 0.0080865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0449691
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0449691
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9396214, 0.9906507, 0.9482200, 0.9905944, -0.0509729, 0.0424306
1: -0.0111462, -0.0026353, -0.0086644, -0.0028184, -0.0083277, 0.0060292
2: 0.0084631, 0.0201587, 0.0090643, 0.0184678, -0.0100048, 0.0110944
3: -0.0089945, 0.0047056, -0.0088326, -0.0001013, -0.0088932, 0.0135382
4: -0.0036628, 0.0057258, -0.0010201, 0.0054624, -0.0091252, 0.0067459
5: 0.0102665, 0.0535832, 0.0103600, 0.0459506, -0.0356841, 0.0432233
6: -0.0079057, 0.0047750, -0.0061832, 0.0026183, -0.0105240, 0.0109582
7: -0.0179735, -0.0039951, -0.0154537, -0.0051693, -0.0128042, 0.0114586
8: -0.0082514, 0.0129347, -0.0069760, 0.0094258, -0.0176772, 0.0199107
9: 0.0000666, 0.0120237, 0.0012617, 0.0094430, -0.0093764, 0.0107621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0450747
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0450747
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9485362, 0.9904261, 0.9366314, 0.9907673, -0.0422311, 0.0537946
1: -0.0085732, -0.0028252, -0.0120091, -0.0025716, -0.0060016, 0.0091839
2: 0.0092866, 0.0184057, 0.0080459, 0.0207467, -0.0114601, 0.0103598
3: -0.0088266, -0.0002781, -0.0090508, 0.0063771, -0.0152037, 0.0087728
4: -0.0009229, 0.0054527, -0.0045817, 0.0058174, -0.0067403, 0.0100344
5: 0.0106395, 0.0456699, 0.0100728, 0.0562372, -0.0455977, 0.0355972
6: -0.0061199, 0.0025390, -0.0085046, 0.0055249, -0.0116447, 0.0110436
7: -0.0153611, -0.0052125, -0.0188496, -0.0035868, -0.0117742, 0.0136372
8: -0.0069291, 0.0092967, -0.0086949, 0.0141548, -0.0210839, 0.0179916
9: 0.0013736, 0.0093481, -0.0003839, 0.0129211, -0.0115475, 0.0097320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0445127
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0445127
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9396214, 0.9906507, 0.9366314, 0.9907673, -0.0511459, 0.0540192
1: -0.0111462, -0.0026353, -0.0120091, -0.0025716, -0.0085746, 0.0093738
2: 0.0084631, 0.0201587, 0.0080459, 0.0207467, -0.0122836, 0.0121128
3: -0.0089945, 0.0047056, -0.0090508, 0.0063771, -0.0153716, 0.0137565
4: -0.0036628, 0.0057258, -0.0045817, 0.0058174, -0.0094802, 0.0103075
5: 0.0102665, 0.0535832, 0.0100728, 0.0562372, -0.0459707, 0.0435105
6: -0.0079057, 0.0047750, -0.0085046, 0.0055249, -0.0134306, 0.0132796
7: -0.0179735, -0.0039951, -0.0188496, -0.0035868, -0.0143866, 0.0148545
8: -0.0082514, 0.0129347, -0.0086949, 0.0141548, -0.0224062, 0.0216295
9: 0.0000666, 0.0120237, -0.0003839, 0.0129211, -0.0128544, 0.0124076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9482200, 0.9905944, 0.9396214, 0.9906507, -0.0424306, 0.0509729
1: -0.0086644, -0.0028184, -0.0111462, -0.0026353, -0.0060292, 0.0083277
2: 0.0090643, 0.0184678, 0.0084631, 0.0201587, -0.0110944, 0.0100048
3: -0.0088326, -0.0001013, -0.0089945, 0.0047056, -0.0135382, 0.0088932
4: -0.0010201, 0.0054624, -0.0036628, 0.0057258, -0.0067459, 0.0091252
5: 0.0103600, 0.0459506, 0.0102665, 0.0535832, -0.0432233, 0.0356841
6: -0.0061832, 0.0026183, -0.0079057, 0.0047750, -0.0109582, 0.0105240
7: -0.0154537, -0.0051693, -0.0179735, -0.0039951, -0.0114586, 0.0128042
8: -0.0069760, 0.0094258, -0.0082514, 0.0129347, -0.0199107, 0.0176772
9: 0.0012617, 0.0094430, 0.0000666, 0.0120237, -0.0107621, 0.0093764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450375, upper bound: 0.0448429
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450375, upper bound: 0.0448429
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9396214, 0.9906507, 0.9392697, 0.9908380, -0.0512165, 0.0513810
1: -0.0111462, -0.0026353, -0.0112477, -0.0026278, -0.0085184, 0.0086124
2: 0.0084631, 0.0201587, 0.0084140, 0.0202279, -0.0117648, 0.0117447
3: -0.0089945, 0.0047056, -0.0090012, 0.0049022, -0.0138968, 0.0137068
4: -0.0036628, 0.0057258, -0.0037709, 0.0057366, -0.0093994, 0.0094967
5: 0.0102665, 0.0535832, 0.0099553, 0.0538954, -0.0436289, 0.0436279
6: -0.0079057, 0.0047750, -0.0079761, 0.0048632, -0.0127689, 0.0127511
7: -0.0179735, -0.0039951, -0.0180765, -0.0039471, -0.0140264, 0.0140814
8: -0.0082514, 0.0129347, -0.0083036, 0.0130782, -0.0213296, 0.0212382
9: 0.0000666, 0.0120237, 0.0000137, 0.0121293, -0.0120626, 0.0120101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448708, upper bound: 0.0450394
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448708, upper bound: 0.0450394
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9485362, 0.9904261, 0.9277400, 0.9910126, -0.0424764, 0.0626861
1: -0.0085732, -0.0028252, -0.0145753, -0.0023822, -0.0061910, 0.0117501
2: 0.0092866, 0.0184057, 0.0068053, 0.0224951, -0.0132086, 0.0116004
3: -0.0088266, -0.0002781, -0.0092183, 0.0113477, -0.0201743, 0.0089403
4: -0.0009229, 0.0054527, -0.0073144, 0.0060898, -0.0070127, 0.0127671
5: 0.0106395, 0.0456699, 0.0094686, 0.0641297, -0.0534902, 0.0362013
6: -0.0061199, 0.0025390, -0.0102857, 0.0077549, -0.0138748, 0.0128248
7: -0.0153611, -0.0052125, -0.0214552, -0.0023727, -0.0129884, 0.0162427
8: -0.0069291, 0.0092967, -0.0100137, 0.0177832, -0.0247123, 0.0193104
9: 0.0013736, 0.0093481, -0.0017236, 0.0155896, -0.0142161, 0.0110717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449915, upper bound: 0.0443866
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449915, upper bound: 0.0443866
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9396214, 0.9906507, 0.9277400, 0.9910126, -0.0513911, 0.0629107
1: -0.0111462, -0.0026353, -0.0145753, -0.0023822, -0.0087640, 0.0119400
2: 0.0084631, 0.0201587, 0.0068053, 0.0224951, -0.0140320, 0.0133534
3: -0.0089945, 0.0047056, -0.0092183, 0.0113477, -0.0203422, 0.0139240
4: -0.0036628, 0.0057258, -0.0073144, 0.0060898, -0.0097526, 0.0130402
5: 0.0102665, 0.0535832, 0.0094686, 0.0641297, -0.0538632, 0.0441146
6: -0.0079057, 0.0047750, -0.0102857, 0.0077549, -0.0156606, 0.0150607
7: -0.0179735, -0.0039951, -0.0214552, -0.0023727, -0.0156008, 0.0174600
8: -0.0082514, 0.0129347, -0.0100137, 0.0177832, -0.0260346, 0.0229484
9: 0.0000666, 0.0120237, -0.0017236, 0.0155896, -0.0155230, 0.0137473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448133, upper bound: 0.0445754
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448133, upper bound: 0.0445754
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9366314, 0.9907673, 0.9485362, 0.9904261, -0.0537946, 0.0422311
1: -0.0120091, -0.0025716, -0.0085732, -0.0028252, -0.0091839, 0.0060016
2: 0.0080459, 0.0207467, 0.0092866, 0.0184057, -0.0103598, 0.0114601
3: -0.0090508, 0.0063771, -0.0088266, -0.0002781, -0.0087728, 0.0152037
4: -0.0045817, 0.0058174, -0.0009229, 0.0054527, -0.0100344, 0.0067403
5: 0.0100728, 0.0562372, 0.0106395, 0.0456699, -0.0355972, 0.0455977
6: -0.0085046, 0.0055249, -0.0061199, 0.0025390, -0.0110436, 0.0116447
7: -0.0188496, -0.0035868, -0.0153611, -0.0052125, -0.0136372, 0.0117742
8: -0.0086949, 0.0141548, -0.0069291, 0.0092967, -0.0179916, 0.0210839
9: -0.0003839, 0.0129211, 0.0013736, 0.0093481, -0.0097320, 0.0115475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470607
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470607
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9485362, 0.9904261, -0.0626861, 0.0424764
1: -0.0145753, -0.0023822, -0.0085732, -0.0028252, -0.0117501, 0.0061910
2: 0.0068053, 0.0224951, 0.0092866, 0.0184057, -0.0116004, 0.0132086
3: -0.0092183, 0.0113477, -0.0088266, -0.0002781, -0.0089403, 0.0201743
4: -0.0073144, 0.0060898, -0.0009229, 0.0054527, -0.0127671, 0.0070127
5: 0.0094686, 0.0641297, 0.0106395, 0.0456699, -0.0362013, 0.0534902
6: -0.0102857, 0.0077549, -0.0061199, 0.0025390, -0.0128248, 0.0138748
7: -0.0214552, -0.0023727, -0.0153611, -0.0052125, -0.0162427, 0.0129884
8: -0.0100137, 0.0177832, -0.0069291, 0.0092967, -0.0193104, 0.0247123
9: -0.0017236, 0.0155896, 0.0013736, 0.0093481, -0.0110717, 0.0142161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470607
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0476103
time: 1.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9369445, 0.9906062, 0.9366314, 0.9907673, -0.0538228, 0.0539748
1: -0.0119187, -0.0025783, -0.0120091, -0.0025716, -0.0093471, 0.0094308
2: 0.0080896, 0.0206851, 0.0080459, 0.0207467, -0.0126571, 0.0126392
3: -0.0090450, 0.0062021, -0.0090508, 0.0063771, -0.0154220, 0.0152530
4: -0.0044855, 0.0058078, -0.0045817, 0.0058174, -0.0103029, 0.0103895
5: 0.0103402, 0.0559594, 0.0100728, 0.0562372, -0.0458970, 0.0458866
6: -0.0084419, 0.0054464, -0.0085046, 0.0055249, -0.0139668, 0.0139510
7: -0.0187579, -0.0036296, -0.0188496, -0.0035868, -0.0151711, 0.0152200
8: -0.0086484, 0.0140271, -0.0086949, 0.0141548, -0.0228032, 0.0227219
9: -0.0003367, 0.0128271, -0.0003839, 0.0129211, -0.0132578, 0.0132110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0445656
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0445656
time: 1.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9280811, 0.9908296, 0.9366314, 0.9907673, -0.0626862, 0.0541981
1: -0.0144768, -0.0023895, -0.0120091, -0.0025716, -0.0119052, 0.0096196
2: 0.0068529, 0.0224280, 0.0080459, 0.0207467, -0.0138938, 0.0143821
3: -0.0092119, 0.0111570, -0.0090508, 0.0063771, -0.0155890, 0.0202078
4: -0.0072095, 0.0060794, -0.0045817, 0.0058174, -0.0130269, 0.0106611
5: 0.0095389, 0.0638269, 0.0100728, 0.0562372, -0.0466983, 0.0537541
6: -0.0102174, 0.0076694, -0.0085046, 0.0055249, -0.0157423, 0.0161740
7: -0.0213552, -0.0024193, -0.0188496, -0.0035868, -0.0177683, 0.0164303
8: -0.0099631, 0.0176440, -0.0086949, 0.0141548, -0.0241179, 0.0263388
9: -0.0016722, 0.0154872, -0.0003839, 0.0129211, -0.0145933, 0.0158711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446687
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446687
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9366314, 0.9907673, 0.9396214, 0.9906507, -0.0540192, 0.0511459
1: -0.0120091, -0.0025716, -0.0111462, -0.0026353, -0.0093738, 0.0085746
2: 0.0080459, 0.0207467, 0.0084631, 0.0201587, -0.0121128, 0.0122836
3: -0.0090508, 0.0063771, -0.0089945, 0.0047056, -0.0137565, 0.0153716
4: -0.0045817, 0.0058174, -0.0036628, 0.0057258, -0.0103075, 0.0094802
5: 0.0100728, 0.0562372, 0.0102665, 0.0535832, -0.0435105, 0.0459707
6: -0.0085046, 0.0055249, -0.0079057, 0.0047750, -0.0132796, 0.0134306
7: -0.0188496, -0.0035868, -0.0179735, -0.0039951, -0.0148545, 0.0143866
8: -0.0086949, 0.0141548, -0.0082514, 0.0129347, -0.0216295, 0.0224062
9: -0.0003839, 0.0129211, 0.0000666, 0.0120237, -0.0124076, 0.0128544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0445685, upper bound: 0.0447799
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0445685, upper bound: 0.0447799
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9277400, 0.9910126, 0.9396214, 0.9906507, -0.0629107, 0.0513911
1: -0.0145753, -0.0023822, -0.0111462, -0.0026353, -0.0119400, 0.0087640
2: 0.0068053, 0.0224951, 0.0084631, 0.0201587, -0.0133534, 0.0140320
3: -0.0092183, 0.0113477, -0.0089945, 0.0047056, -0.0139240, 0.0203422
4: -0.0073144, 0.0060898, -0.0036628, 0.0057258, -0.0130402, 0.0097526
5: 0.0094686, 0.0641297, 0.0102665, 0.0535832, -0.0441146, 0.0538632
6: -0.0102857, 0.0077549, -0.0079057, 0.0047750, -0.0150607, 0.0156606
7: -0.0214552, -0.0023727, -0.0179735, -0.0039951, -0.0174600, 0.0156008
8: -0.0100137, 0.0177832, -0.0082514, 0.0129347, -0.0229484, 0.0260346
9: -0.0017236, 0.0155896, 0.0000666, 0.0120237, -0.0137473, 0.0155230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0444107, upper bound: 0.0449944
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0444107, upper bound: 0.0449944
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9366314, 0.9907673, 0.9280811, 0.9908296, -0.0541981, 0.0626862
1: -0.0120091, -0.0025716, -0.0144768, -0.0023895, -0.0096196, 0.0119052
2: 0.0080459, 0.0207467, 0.0068529, 0.0224280, -0.0143821, 0.0138938
3: -0.0090508, 0.0063771, -0.0092119, 0.0111570, -0.0202078, 0.0155890
4: -0.0045817, 0.0058174, -0.0072095, 0.0060794, -0.0106611, 0.0130269
5: 0.0100728, 0.0562372, 0.0095389, 0.0638269, -0.0537541, 0.0466983
6: -0.0085046, 0.0055249, -0.0102174, 0.0076694, -0.0161740, 0.0157423
7: -0.0188496, -0.0035868, -0.0213552, -0.0024193, -0.0164303, 0.0177683
8: -0.0086949, 0.0141548, -0.0099631, 0.0176440, -0.0263388, 0.0241179
9: -0.0003839, 0.0129211, -0.0016722, 0.0154872, -0.0158711, 0.0145933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0445683, upper bound: 0.0444178
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0445683, upper bound: 0.0444178
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9280811, 0.9908296, 0.9277400, 0.9910126, -0.0629315, 0.0630896
1: -0.0144768, -0.0023895, -0.0145753, -0.0023822, -0.0120946, 0.0121858
2: 0.0068529, 0.0224280, 0.0068053, 0.0224951, -0.0156422, 0.0156227
3: -0.0092119, 0.0111570, -0.0092183, 0.0113477, -0.0205596, 0.0203753
4: -0.0072095, 0.0060794, -0.0073144, 0.0060898, -0.0132993, 0.0133938
5: 0.0095389, 0.0638269, 0.0094686, 0.0641297, -0.0545908, 0.0543583
6: -0.0102174, 0.0076694, -0.0102857, 0.0077549, -0.0179723, 0.0179551
7: -0.0213552, -0.0024193, -0.0214552, -0.0023727, -0.0189825, 0.0190359
8: -0.0099631, 0.0176440, -0.0100137, 0.0177832, -0.0277462, 0.0276576
9: -0.0016722, 0.0154872, -0.0017236, 0.0155896, -0.0172618, 0.0172109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0444105, upper bound: 0.0446183
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0444105, upper bound: 0.0446183
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9485362, 0.9904261, 0.9285241, 0.9905008, -0.0419646, 0.0619020
1: -0.0085732, -0.0028252, -0.0143490, -0.0023989, -0.0061743, 0.0115238
2: 0.0092866, 0.0184057, 0.0069147, 0.0223409, -0.0130544, 0.0114910
3: -0.0088266, -0.0002781, -0.0092036, 0.0109093, -0.0197359, 0.0089255
4: -0.0009229, 0.0054527, -0.0070734, 0.0060658, -0.0069887, 0.0125260
5: 0.0106395, 0.0456699, 0.0096302, 0.0634336, -0.0527942, 0.0360398
6: -0.0061199, 0.0025390, -0.0101287, 0.0075583, -0.0136781, 0.0126677
7: -0.0153611, -0.0052125, -0.0212254, -0.0024798, -0.0128813, 0.0160129
8: -0.0069291, 0.0092967, -0.0098973, 0.0174632, -0.0243923, 0.0191941
9: 0.0013736, 0.0093481, -0.0016055, 0.0153543, -0.0139807, 0.0109536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9485362, 0.9904261, 0.9197506, 0.9907448, -0.0422086, 0.0706755
1: -0.0085732, -0.0028252, -0.0168811, -0.0022120, -0.0063612, 0.0140560
2: 0.0092866, 0.0184057, 0.0056905, 0.0240662, -0.0147796, 0.0127151
3: -0.0088266, -0.0002781, -0.0093688, 0.0158140, -0.0246406, 0.0090908
4: -0.0009229, 0.0054527, -0.0097698, 0.0063346, -0.0072575, 0.0152225
5: 0.0106395, 0.0456699, 0.0078221, 0.0712213, -0.0605818, 0.0378478
6: -0.0061199, 0.0025390, -0.0118862, 0.0097587, -0.0158786, 0.0144252
7: -0.0153611, -0.0052125, -0.0237963, -0.0012817, -0.0140793, 0.0185839
8: -0.0069291, 0.0092967, -0.0111987, 0.0210434, -0.0279725, 0.0204954
9: 0.0013736, 0.0093481, -0.0029274, 0.0179875, -0.0166139, 0.0122755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9485362, 0.9904261, 0.9185466, 0.9906401, -0.0421039, 0.0718794
1: -0.0085732, -0.0028252, -0.0172287, -0.0021864, -0.0063868, 0.0144035
2: 0.0092866, 0.0184057, 0.0055225, 0.0243029, -0.0150164, 0.0128832
3: -0.0088266, -0.0002781, -0.0093915, 0.0164871, -0.0253137, 0.0091135
4: -0.0009229, 0.0054527, -0.0101399, 0.0063715, -0.0072944, 0.0155925
5: 0.0106395, 0.0456699, 0.0075740, 0.0722903, -0.0616508, 0.0380959
6: -0.0061199, 0.0025390, -0.0121274, 0.0100608, -0.0161806, 0.0146664
7: -0.0153611, -0.0052125, -0.0241492, -0.0011173, -0.0142437, 0.0189367
8: -0.0069291, 0.0092967, -0.0113773, 0.0215348, -0.0284639, 0.0206740
9: 0.0013736, 0.0093481, -0.0031089, 0.0183488, -0.0169753, 0.0124570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9485362, 0.9904261, 0.9090129, 0.9908924, -0.0423562, 0.0814132
1: -0.0085732, -0.0028252, -0.0199802, -0.0019833, -0.0065899, 0.0171551
2: 0.0092866, 0.0184057, 0.0041923, 0.0261777, -0.0168911, 0.0142134
3: -0.0088266, -0.0002781, -0.0095711, 0.0218167, -0.0306433, 0.0092930
4: -0.0009229, 0.0054527, -0.0130699, 0.0066636, -0.0075865, 0.0185226
5: 0.0106395, 0.0456699, 0.0056093, 0.0807528, -0.0701133, 0.0400606
6: -0.0061199, 0.0025390, -0.0140371, 0.0124519, -0.0185718, 0.0165762
7: -0.0153611, -0.0052125, -0.0269429, 0.0001845, -0.0155456, 0.0217305
8: -0.0069291, 0.0092967, -0.0127913, 0.0254253, -0.0323544, 0.0220881
9: 0.0013736, 0.0093481, -0.0045454, 0.0212101, -0.0198366, 0.0138935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9396214, 0.9906507, 0.9285241, 0.9905008, -0.0508794, 0.0621266
1: -0.0111462, -0.0026353, -0.0143490, -0.0023989, -0.0087473, 0.0117137
2: 0.0084631, 0.0201587, 0.0069147, 0.0223409, -0.0138778, 0.0132440
3: -0.0089945, 0.0047056, -0.0092036, 0.0109093, -0.0199038, 0.0139092
4: -0.0036628, 0.0057258, -0.0070734, 0.0060658, -0.0097286, 0.0127992
5: 0.0102665, 0.0535832, 0.0096302, 0.0634336, -0.0531671, 0.0439531
6: -0.0079057, 0.0047750, -0.0101287, 0.0075583, -0.0154640, 0.0149036
7: -0.0179735, -0.0039951, -0.0212254, -0.0024798, -0.0154937, 0.0172302
8: -0.0082514, 0.0129347, -0.0098973, 0.0174632, -0.0257146, 0.0228320
9: 0.0000666, 0.0120237, -0.0016055, 0.0153543, -0.0152876, 0.0136292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448038, upper bound: 0.0448937
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448038, upper bound: 0.0448937
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9396214, 0.9906507, 0.9197506, 0.9907448, -0.0511233, 0.0709001
1: -0.0111462, -0.0026353, -0.0168811, -0.0022120, -0.0089341, 0.0142458
2: 0.0084631, 0.0201587, 0.0056905, 0.0240662, -0.0156031, 0.0144682
3: -0.0089945, 0.0047056, -0.0093688, 0.0158140, -0.0248085, 0.0140745
4: -0.0036628, 0.0057258, -0.0097698, 0.0063346, -0.0099974, 0.0154956
5: 0.0102665, 0.0535832, 0.0078221, 0.0712213, -0.0609548, 0.0457611
6: -0.0079057, 0.0047750, -0.0118862, 0.0097587, -0.0176644, 0.0166611
7: -0.0179735, -0.0039951, -0.0237963, -0.0012817, -0.0166917, 0.0198012
8: -0.0082514, 0.0129347, -0.0111987, 0.0210434, -0.0292948, 0.0241333
9: 0.0000666, 0.0120237, -0.0029274, 0.0179875, -0.0179208, 0.0149512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448230, upper bound: 0.0448949
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0448230, upper bound: 0.0448949
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9396214, 0.9906507, 0.9185466, 0.9906401, -0.0510187, 0.0721040
1: -0.0111462, -0.0026353, -0.0172287, -0.0021864, -0.0089598, 0.0145934
2: 0.0084631, 0.0201587, 0.0055225, 0.0243029, -0.0158398, 0.0146362
3: -0.0089945, 0.0047056, -0.0093915, 0.0164871, -0.0254816, 0.0140971
4: -0.0036628, 0.0057258, -0.0101399, 0.0063715, -0.0100343, 0.0158657
5: 0.0102665, 0.0535832, 0.0075740, 0.0722903, -0.0620238, 0.0460092
6: -0.0079057, 0.0047750, -0.0121274, 0.0100608, -0.0179665, 0.0169023
7: -0.0179735, -0.0039951, -0.0241492, -0.0011173, -0.0168562, 0.0201541
8: -0.0082514, 0.0129347, -0.0113773, 0.0215348, -0.0297862, 0.0243120
9: 0.0000666, 0.0120237, -0.0031089, 0.0183488, -0.0182822, 0.0151326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447349, upper bound: 0.0443126
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447349, upper bound: 0.0443126
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9396214, 0.9906507, 0.9090129, 0.9908924, -0.0512710, 0.0816378
1: -0.0111462, -0.0026353, -0.0199802, -0.0019833, -0.0091629, 0.0173449
2: 0.0084631, 0.0201587, 0.0041923, 0.0261777, -0.0177146, 0.0159664
3: -0.0089945, 0.0047056, -0.0095711, 0.0218167, -0.0308113, 0.0142767
4: -0.0036628, 0.0057258, -0.0130699, 0.0066636, -0.0103264, 0.0187957
5: 0.0102665, 0.0535832, 0.0056093, 0.0807528, -0.0704863, 0.0479739
6: -0.0079057, 0.0047750, -0.0140371, 0.0124519, -0.0203576, 0.0188121
7: -0.0179735, -0.0039951, -0.0269429, 0.0001845, -0.0181580, 0.0229478
8: -0.0082514, 0.0129347, -0.0127913, 0.0254253, -0.0336767, 0.0257260
9: 0.0000666, 0.0120237, -0.0045454, 0.0212101, -0.0211435, 0.0165691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447604, upper bound: 0.0443171
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0447604, upper bound: 0.0443171
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9369445, 0.9906062, 0.9285241, 0.9905008, -0.0535563, 0.0620822
1: -0.0119187, -0.0025783, -0.0143490, -0.0023989, -0.0095198, 0.0117707
2: 0.0080896, 0.0206851, 0.0069147, 0.0223409, -0.0142513, 0.0137704
3: -0.0090450, 0.0062021, -0.0092036, 0.0109093, -0.0199543, 0.0154057
4: -0.0044855, 0.0058078, -0.0070734, 0.0060658, -0.0105513, 0.0128812
5: 0.0103402, 0.0559594, 0.0096302, 0.0634336, -0.0530934, 0.0463292
6: -0.0084419, 0.0054464, -0.0101287, 0.0075583, -0.0160002, 0.0155750
7: -0.0187579, -0.0036296, -0.0212254, -0.0024798, -0.0162781, 0.0175958
8: -0.0086484, 0.0140271, -0.0098973, 0.0174632, -0.0261116, 0.0239244
9: -0.0003367, 0.0128271, -0.0016055, 0.0153543, -0.0156910, 0.0144326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9369445, 0.9906062, 0.9197506, 0.9907448, -0.0538003, 0.0708557
1: -0.0119187, -0.0025783, -0.0168811, -0.0022120, -0.0097067, 0.0143029
2: 0.0080896, 0.0206851, 0.0056905, 0.0240662, -0.0159766, 0.0149946
3: -0.0090450, 0.0062021, -0.0093688, 0.0158140, -0.0248589, 0.0155709
4: -0.0044855, 0.0058078, -0.0097698, 0.0063346, -0.0108201, 0.0155776
5: 0.0103402, 0.0559594, 0.0078221, 0.0712213, -0.0608811, 0.0481372
6: -0.0084419, 0.0054464, -0.0118862, 0.0097587, -0.0182007, 0.0173325
7: -0.0187579, -0.0036296, -0.0237963, -0.0012817, -0.0174762, 0.0201668
8: -0.0086484, 0.0140271, -0.0111987, 0.0210434, -0.0296919, 0.0252257
9: -0.0003367, 0.0128271, -0.0029274, 0.0179875, -0.0183242, 0.0157546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
time: 1.56 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9369445, 0.9906062, 0.9185466, 0.9906401, -0.0536956, 0.0720596
1: -0.0119187, -0.0025783, -0.0172287, -0.0021864, -0.0097324, 0.0146504
2: 0.0080896, 0.0206851, 0.0055225, 0.0243029, -0.0162133, 0.0151626
3: -0.0090450, 0.0062021, -0.0093915, 0.0164871, -0.0255320, 0.0155936
4: -0.0044855, 0.0058078, -0.0101399, 0.0063715, -0.0108570, 0.0159477
5: 0.0103402, 0.0559594, 0.0075740, 0.0722903, -0.0619500, 0.0483854
6: -0.0084419, 0.0054464, -0.0121274, 0.0100608, -0.0185027, 0.0175737
7: -0.0187579, -0.0036296, -0.0241492, -0.0011173, -0.0176406, 0.0205196
8: -0.0086484, 0.0140271, -0.0113773, 0.0215348, -0.0301832, 0.0254043
9: -0.0003367, 0.0128271, -0.0031089, 0.0183488, -0.0186855, 0.0159360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
time: 1.51 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9369445, 0.9906062, 0.9090129, 0.9908924, -0.0539479, 0.0815934
1: -0.0119187, -0.0025783, -0.0199802, -0.0019833, -0.0099354, 0.0174020
2: 0.0080896, 0.0206851, 0.0041923, 0.0261777, -0.0180881, 0.0164928
3: -0.0090450, 0.0062021, -0.0095711, 0.0218167, -0.0308617, 0.0157732
4: -0.0044855, 0.0058078, -0.0130699, 0.0066636, -0.0111491, 0.0188778
5: 0.0103402, 0.0559594, 0.0056093, 0.0807528, -0.0704126, 0.0503501
6: -0.0084419, 0.0054464, -0.0140371, 0.0124519, -0.0208938, 0.0194835
7: -0.0187579, -0.0036296, -0.0269429, 0.0001845, -0.0189424, 0.0233133
8: -0.0086484, 0.0140271, -0.0127913, 0.0254253, -0.0340737, 0.0268184
9: -0.0003367, 0.0128271, -0.0045454, 0.0212101, -0.0215469, 0.0173725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9280811, 0.9908296, 0.9285241, 0.9905008, -0.0624197, 0.0623055
1: -0.0144768, -0.0023895, -0.0143490, -0.0023989, -0.0120779, 0.0119595
2: 0.0068529, 0.0224280, 0.0069147, 0.0223409, -0.0154880, 0.0155133
3: -0.0092119, 0.0111570, -0.0092036, 0.0109093, -0.0201212, 0.0203605
4: -0.0072095, 0.0060794, -0.0070734, 0.0060658, -0.0132753, 0.0131527
5: 0.0095389, 0.0638269, 0.0096302, 0.0634336, -0.0538947, 0.0541967
6: -0.0102174, 0.0076694, -0.0101287, 0.0075583, -0.0177757, 0.0177980
7: -0.0213552, -0.0024193, -0.0212254, -0.0024798, -0.0188754, 0.0188061
8: -0.0099631, 0.0176440, -0.0098973, 0.0174632, -0.0274263, 0.0275413
9: -0.0016722, 0.0154872, -0.0016055, 0.0153543, -0.0170265, 0.0170927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443242, upper bound: 0.0448489
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443242, upper bound: 0.0448489
time: 1.70 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9280811, 0.9908296, 0.9197506, 0.9907448, -0.0626637, 0.0710790
1: -0.0144768, -0.0023895, -0.0168811, -0.0022120, -0.0122648, 0.0144917
2: 0.0068529, 0.0224280, 0.0056905, 0.0240662, -0.0172133, 0.0167375
3: -0.0092119, 0.0111570, -0.0093688, 0.0158140, -0.0250259, 0.0205258
4: -0.0072095, 0.0060794, -0.0097698, 0.0063346, -0.0135441, 0.0158492
5: 0.0095389, 0.0638269, 0.0078221, 0.0712213, -0.0616824, 0.0560048
6: -0.0102174, 0.0076694, -0.0118862, 0.0097587, -0.0199761, 0.0195555
7: -0.0213552, -0.0024193, -0.0237963, -0.0012817, -0.0200734, 0.0213771
8: -0.0099631, 0.0176440, -0.0111987, 0.0210434, -0.0310065, 0.0288426
9: -0.0016722, 0.0154872, -0.0029274, 0.0179875, -0.0196597, 0.0184147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443472, upper bound: 0.0448501
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443472, upper bound: 0.0448501
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9280811, 0.9908296, 0.9185466, 0.9906401, -0.0625590, 0.0722830
1: -0.0144768, -0.0023895, -0.0172287, -0.0021864, -0.0122905, 0.0148392
2: 0.0068529, 0.0224280, 0.0055225, 0.0243029, -0.0174501, 0.0169055
3: -0.0092119, 0.0111570, -0.0093915, 0.0164871, -0.0256990, 0.0205485
4: -0.0072095, 0.0060794, -0.0101399, 0.0063715, -0.0135810, 0.0162192
5: 0.0095389, 0.0638269, 0.0075740, 0.0722903, -0.0627514, 0.0562529
6: -0.0102174, 0.0076694, -0.0121274, 0.0100608, -0.0202782, 0.0197967
7: -0.0213552, -0.0024193, -0.0241492, -0.0011173, -0.0202379, 0.0217299
8: -0.0099631, 0.0176440, -0.0113773, 0.0215348, -0.0314979, 0.0290212
9: -0.0016722, 0.0154872, -0.0031089, 0.0183488, -0.0200210, 0.0185961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443236, upper bound: 0.0443570
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0443236, upper bound: 0.0443570
time: 1.10 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.84 seconds
IS_A1_B1_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0449691
IS_A1_B1_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0449691
IS_A1_B1_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0450747
IS_A1_B1_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0471855, upper bound: 0.0450747
IS_A1_B1_A1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0445127
IS_A1_B1_A1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0445127
IS_A1_B1_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
IS_A1_B1_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0470071, upper bound: 0.0446116
IS_A1_B1_A1_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0450375, upper bound: 0.0448429
IS_A1_B1_A1_B2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0450375, upper bound: 0.0448429
IS_A1_B1_A1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0448708, upper bound: 0.0450394
IS_A1_B1_A1_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0448708, upper bound: 0.0450394
IS_A1_B1_A1_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0449915, upper bound: 0.0443866
IS_A1_B1_A1_B2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0449915, upper bound: 0.0443866
IS_A1_B1_A1_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0448133, upper bound: 0.0445754
IS_A1_B1_A1_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0448133, upper bound: 0.0445754
IS_A1_B1_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470607
IS_A1_B1_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470607
IS_A1_B1_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0470607
IS_A1_B1_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0445127, upper bound: 0.0476103
IS_A1_B1_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0445656
IS_A1_B1_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0445656
IS_A1_B1_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446687
IS_A1_B1_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0467338, upper bound: 0.0446687
IS_A1_B1_A2_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0445685, upper bound: 0.0447799
IS_A1_B1_A2_B2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0445685, upper bound: 0.0447799
IS_A1_B1_A2_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0444107, upper bound: 0.0449944
IS_A1_B1_A2_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0444107, upper bound: 0.0449944
IS_A1_B1_A2_B2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0445683, upper bound: 0.0444178
IS_A1_B1_A2_B2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0445683, upper bound: 0.0444178
IS_A1_B1_A2_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0444105, upper bound: 0.0446183
IS_A1_B1_A2_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0444105, upper bound: 0.0446183
IS_A1_B2_A1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
IS_A1_B2_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
IS_A1_B2_A1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
IS_A1_B2_A1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0471528, upper bound: 0.0448322
IS_A1_B2_A1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
IS_A1_B2_A1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
IS_A1_B2_A1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
IS_A1_B2_A1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0469520, upper bound: 0.0442552
IS_A1_B2_A1_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0448038, upper bound: 0.0448937
IS_A1_B2_A1_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0448038, upper bound: 0.0448937
IS_A1_B2_A1_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0448230, upper bound: 0.0448949
IS_A1_B2_A1_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0448230, upper bound: 0.0448949
IS_A1_B2_A1_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0447349, upper bound: 0.0443126
IS_A1_B2_A1_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0447349, upper bound: 0.0443126
IS_A1_B2_A1_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0447604, upper bound: 0.0443171
IS_A1_B2_A1_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0447604, upper bound: 0.0443171
IS_A1_B2_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
IS_A1_B2_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
IS_A1_B2_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
IS_A1_B2_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0467036, upper bound: 0.0447965
IS_A1_B2_A2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
IS_A1_B2_A2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
IS_A1_B2_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
IS_A1_B2_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0466600, upper bound: 0.0443075
IS_A1_B2_A2_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0443242, upper bound: 0.0448489
IS_A1_B2_A2_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0443242, upper bound: 0.0448489
IS_A1_B2_A2_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0443472, upper bound: 0.0448501
IS_A1_B2_A2_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0443472, upper bound: 0.0448501
IS_A1_B2_A2_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0443236, upper bound: 0.0443570
IS_A1_B2_A2_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 0, lower bound: -0.0443236, upper bound: 0.0443570
IS_A1_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0466329, upper bound: 0.0444164
IS_A2_B1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0471528
IS_A2_B1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0476755
IS_A2_B1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0469520
IS_A2_B1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
IS_A2_B1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0471153
IS_A2_B1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0448322, upper bound: 0.0476755
IS_A2_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0469132
IS_A2_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0475262
IS_A2_B1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0470147, upper bound: 0.0448382
IS_A2_B1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0448068, upper bound: 0.0469904
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0467822
IS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0467822
IS_A2_B1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0469761, upper bound: 0.0449315
IS_A2_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0469761, upper bound: 0.0449317
IS_A2_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0474100
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0474100
IS_A2_B2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0467036
IS_A2_B2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0472362
IS_A2_B2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0466600
IS_A2_B2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
IS_A2_B2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0466722
IS_A2_B2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0447965, upper bound: 0.0472362
IS_A2_B2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0466329
IS_A2_B2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442552, upper bound: 0.0472107
IS_A2_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0468161, upper bound: 0.0442667
IS_A2_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0468161, upper bound: 0.0442667
IS_A2_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0464751, upper bound: 0.0443010
IS_A2_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0442322, upper bound: 0.0464602
IS_A2_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0467791, upper bound: 0.0443563
IS_A2_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0467791, upper bound: 0.0443593
IS_A2_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0464524, upper bound: 0.0443987
IS_A2_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -0.0464524, upper bound: 0.0444019

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.52 + 597.07 = 600.58 seconds
