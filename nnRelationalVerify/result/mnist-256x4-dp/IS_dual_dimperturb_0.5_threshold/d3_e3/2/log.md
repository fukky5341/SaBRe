## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01546812


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0080679, 0.0068331, -0.0080679, 0.0068331, -0.0149009, 0.0149009)
1: (0.9888619, 1.0130347, 0.9888619, 1.0130347, -0.0241728, 0.0241728)
2: (-0.0140929, 0.0072175, -0.0140929, 0.0072175, -0.0209244, 0.0209244)
3: (-0.0005442, 0.0059636, -0.0005442, 0.0059636, -0.0065078, 0.0065078)
4: (-0.0087648, 0.0095553, -0.0087648, 0.0095553, -0.0183201, 0.0183201)
5: (-0.0028948, 0.0110775, -0.0028948, 0.0110775, -0.0139723, 0.0139723)
6: (-0.0119045, 0.0037662, -0.0119045, 0.0037662, -0.0156708, 0.0156708)
7: (-0.0115885, 0.0010988, -0.0115885, 0.0010988, -0.0126873, 0.0126873)
8: (-0.0149248, 0.0167679, -0.0149248, 0.0167679, -0.0315306, 0.0315306)
9: (-0.0098271, 0.0084186, -0.0098271, 0.0084186, -0.0182457, 0.0182457)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.64 + 2.98 = 4.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0192520, upper bound: 0.0192519

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0189486, upper bound: 0.0190192
time: 2.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0189486, upper bound: 0.0189486
time: 2.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.90 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.90
Output dim: 1, lower bound: -0.0189486, upper bound: 0.0190192
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.90
Output dim: 1, lower bound: -0.0189486, upper bound: 0.0189486

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0072604, 0.0057760, -0.0079159, 0.0066341, -0.0138945, 0.0136919
1: 0.9889029, 1.0114315, 0.9888692, 1.0127330, -0.0238301, 0.0225623
2: -0.0140212, 0.0063812, -0.0140800, 0.0070601, -0.0206576, 0.0200491
3: -0.0003182, 0.0059465, -0.0005017, 0.0059605, -0.0062787, 0.0064482
4: -0.0077020, 0.0094987, -0.0085648, 0.0095451, -0.0172471, 0.0180635
5: -0.0025270, 0.0110375, -0.0028256, 0.0110703, -0.0135973, 0.0138630
6: -0.0102629, 0.0037100, -0.0115956, 0.0037561, -0.0140190, 0.0153056
7: -0.0115597, 0.0005130, -0.0115833, 0.0009885, -0.0125482, 0.0120964
8: -0.0141062, 0.0166738, -0.0147708, 0.0167511, -0.0306852, 0.0312807
9: -0.0097748, 0.0079384, -0.0098177, 0.0083282, -0.0181031, 0.0177561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179976, upper bound: 0.0185002
time: 2.69 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0188297, upper bound: 0.0188965
time: 2.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0070610, 0.0055149, -0.0077359, 0.0063985, -0.0134595, 0.0132508
1: 0.9887944, 1.0110354, 0.9888725, 1.0123757, -0.0235813, 0.0221629
2: -0.0142101, 0.0061747, -0.0140743, 0.0068737, -0.0207044, 0.0198545
3: -0.0002623, 0.0059915, -0.0004513, 0.0059592, -0.0062215, 0.0064428
4: -0.0074395, 0.0096479, -0.0083279, 0.0095406, -0.0169801, 0.0179758
5: -0.0024362, 0.0111430, -0.0027436, 0.0110671, -0.0135033, 0.0138866
6: -0.0098574, 0.0038581, -0.0112297, 0.0037517, -0.0136091, 0.0150878
7: -0.0116357, 0.0003684, -0.0115811, 0.0008580, -0.0124936, 0.0119494
8: -0.0139040, 0.0169219, -0.0145883, 0.0167436, -0.0304868, 0.0313650
9: -0.0099126, 0.0078198, -0.0098136, 0.0082212, -0.0181338, 0.0176333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179976, upper bound: 0.0184301
time: 1.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0188297, upper bound: 0.0188297
time: 2.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.99 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.99
Output dim: 1, lower bound: -0.0179976, upper bound: 0.0185002
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.99
Output dim: 1, lower bound: -0.0188297, upper bound: 0.0188965
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.99
Output dim: 1, lower bound: -0.0179976, upper bound: 0.0184301
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.99
Output dim: 1, lower bound: -0.0188297, upper bound: 0.0188297

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0067429, 0.0050986, -0.0063822, 0.0046263, -0.0113693, 0.0114808
1: 0.9889179, 1.0104041, 0.9885892, 1.0096878, -0.0207698, 0.0218149
2: -0.0139950, 0.0058453, -0.0145672, 0.0054717, -0.0190175, 0.0198848
3: -0.0001733, 0.0059402, -0.0000723, 0.0060767, -0.0062500, 0.0060126
4: -0.0070209, 0.0094779, -0.0065462, 0.0099301, -0.0169511, 0.0160241
5: -0.0022913, 0.0110228, -0.0021270, 0.0113425, -0.0136339, 0.0131498
6: -0.0092108, 0.0036894, -0.0084775, 0.0041381, -0.0133490, 0.0121669
7: -0.0115491, 0.0001377, -0.0117793, -0.0001240, -0.0114252, 0.0119170
8: -0.0135816, 0.0166393, -0.0132159, 0.0173911, -0.0307881, 0.0296878
9: -0.0097557, 0.0076306, -0.0101732, 0.0074161, -0.0171717, 0.0178038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179464, upper bound: 0.0181630
time: 1.82 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179805, upper bound: 0.0184695
time: 1.89 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0072604, 0.0057760, -0.0075067, 0.0060985, -0.0133589, 0.0132827
1: 0.9889029, 1.0114315, 0.9888791, 1.0119207, -0.0230178, 0.0225524
2: -0.0140212, 0.0063812, -0.0140627, 0.0066363, -0.0201678, 0.0200133
3: -0.0003182, 0.0059465, -0.0003871, 0.0059564, -0.0062745, 0.0063336
4: -0.0077020, 0.0094987, -0.0080262, 0.0095313, -0.0172334, 0.0175249
5: -0.0025270, 0.0110375, -0.0026392, 0.0110606, -0.0135876, 0.0136767
6: -0.0102629, 0.0037100, -0.0107637, 0.0037425, -0.0140054, 0.0144737
7: -0.0115597, 0.0005130, -0.0115763, 0.0006917, -0.0122514, 0.0120894
8: -0.0141062, 0.0166738, -0.0143559, 0.0167282, -0.0306624, 0.0308527
9: -0.0097748, 0.0079384, -0.0098050, 0.0080849, -0.0178597, 0.0177434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184315, upper bound: 0.0181601
time: 2.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184315, upper bound: 0.0188964
time: 2.08 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0065253, 0.0048136, -0.0061916, 0.0043768, -0.0109021, 0.0110052
1: 0.9888097, 1.0099717, 0.9885930, 1.0093094, -0.0204997, 0.0213787
2: -0.0141835, 0.0056199, -0.0145604, 0.0052743, -0.0190614, 0.0196764
3: -0.0001124, 0.0059852, -0.0000190, 0.0060751, -0.0061874, 0.0060042
4: -0.0067345, 0.0096269, -0.0062953, 0.0099248, -0.0166592, 0.0159221
5: -0.0021922, 0.0111281, -0.0020402, 0.0113387, -0.0135309, 0.0131684
6: -0.0087683, 0.0038373, -0.0080899, 0.0041328, -0.0129011, 0.0119272
7: -0.0116250, -0.0000202, -0.0117766, -0.0002622, -0.0113627, 0.0117564
8: -0.0133609, 0.0168870, -0.0130226, 0.0173822, -0.0305692, 0.0297546
9: -0.0098932, 0.0075012, -0.0101682, 0.0073027, -0.0171959, 0.0176694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179463, upper bound: 0.0181152
time: 1.92 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179805, upper bound: 0.0184060
time: 2.29 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0070610, 0.0055149, -0.0073312, 0.0058686, -0.0129296, 0.0128461
1: 0.9887944, 1.0110354, 0.9888824, 1.0115720, -0.0227776, 0.0221530
2: -0.0142101, 0.0061747, -0.0140570, 0.0064545, -0.0202189, 0.0198194
3: -0.0002623, 0.0059915, -0.0003380, 0.0059550, -0.0062173, 0.0063295
4: -0.0074395, 0.0096479, -0.0077952, 0.0095268, -0.0169664, 0.0174430
5: -0.0024362, 0.0111430, -0.0025593, 0.0110574, -0.0134936, 0.0137022
6: -0.0098574, 0.0038581, -0.0104067, 0.0037380, -0.0135954, 0.0142648
7: -0.0116357, 0.0003684, -0.0115741, 0.0005644, -0.0122000, 0.0119424
8: -0.0139040, 0.0169219, -0.0141779, 0.0167207, -0.0304639, 0.0309380
9: -0.0099126, 0.0078198, -0.0098009, 0.0079804, -0.0178930, 0.0176206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184301, upper bound: 0.0179975
time: 1.90 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184301, upper bound: 0.0188290
time: 2.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.76 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 1, lower bound: -0.0179464, upper bound: 0.0181630
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 1, lower bound: -0.0179805, upper bound: 0.0184695
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 1, lower bound: -0.0184315, upper bound: 0.0181601
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 1, lower bound: -0.0184315, upper bound: 0.0188964
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 1, lower bound: -0.0179463, upper bound: 0.0181152
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 1, lower bound: -0.0179805, upper bound: 0.0184060
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 1, lower bound: -0.0184301, upper bound: 0.0179975
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 1, lower bound: -0.0184301, upper bound: 0.0188290

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0065870, 0.0048945, -0.0062768, 0.0044883, -0.0110753, 0.0111713
1: 0.9891065, 1.0100946, 0.9892678, 1.0094784, -0.0203720, 0.0208268
2: -0.0136676, 0.0056838, -0.0133864, 0.0053625, -0.0185293, 0.0185551
3: -0.0001297, 0.0058622, -0.0000428, 0.0057951, -0.0059248, 0.0059050
4: -0.0068157, 0.0092192, -0.0064074, 0.0089969, -0.0158126, 0.0156266
5: -0.0022203, 0.0108399, -0.0020790, 0.0106828, -0.0129031, 0.0129189
6: -0.0088939, 0.0034327, -0.0082631, 0.0032123, -0.0121061, 0.0116959
7: -0.0114174, 0.0000246, -0.0113043, -0.0002004, -0.0112170, 0.0113289
8: -0.0134235, 0.0162092, -0.0131090, 0.0158397, -0.0290774, 0.0291446
9: -0.0095168, 0.0075379, -0.0093117, 0.0073534, -0.0168702, 0.0168495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179464, upper bound: 0.0181628
time: 2.49 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179464, upper bound: 0.0181630
time: 1.89 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0067429, 0.0050986, -0.0062607, 0.0044673, -0.0112102, 0.0113593
1: 0.9889179, 1.0104041, 0.9886739, 1.0094464, -0.0205284, 0.0217302
2: -0.0139950, 0.0058453, -0.0144200, 0.0053459, -0.0188658, 0.0196540
3: -0.0001733, 0.0059402, -0.0000383, 0.0060416, -0.0062149, 0.0059785
4: -0.0070209, 0.0094779, -0.0063862, 0.0098138, -0.0168347, 0.0158641
5: -0.0022913, 0.0110228, -0.0020717, 0.0112603, -0.0135516, 0.0130945
6: -0.0092108, 0.0036894, -0.0082304, 0.0040227, -0.0132336, 0.0119198
7: -0.0115491, 0.0001377, -0.0117201, -0.0002121, -0.0113370, 0.0118578
8: -0.0135816, 0.0166393, -0.0130927, 0.0171978, -0.0305773, 0.0295627
9: -0.0097557, 0.0076306, -0.0100658, 0.0073438, -0.0170995, 0.0176964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178274, upper bound: 0.0181153
time: 2.15 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175848, upper bound: 0.0180893
time: 1.67 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0056889, 0.0037187, -0.0075067, 0.0060985, -0.0117874, 0.0112254
1: 0.9886239, 1.0083110, 0.9888791, 1.0119207, -0.0232968, 0.0194319
2: -0.0145068, 0.0047537, -0.0140627, 0.0066363, -0.0206277, 0.0183767
3: 0.0001218, 0.0060623, -0.0003871, 0.0059564, -0.0058346, 0.0064494
4: -0.0056336, 0.0098824, -0.0080262, 0.0095313, -0.0151650, 0.0179086
5: -0.0018113, 0.0113087, -0.0026392, 0.0110606, -0.0128719, 0.0139480
6: -0.0070679, 0.0040908, -0.0107637, 0.0037425, -0.0108104, 0.0148545
7: -0.0117550, -0.0006269, -0.0115763, 0.0006917, -0.0124467, 0.0109495
8: -0.0125130, 0.0173118, -0.0143559, 0.0167282, -0.0290663, 0.0314951
9: -0.0101291, 0.0070037, -0.0098050, 0.0080849, -0.0182140, 0.0168088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178403, upper bound: 0.0180550
time: 1.89 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179765, upper bound: 0.0181392
time: 2.29 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0068428, 0.0052293, -0.0075067, 0.0060985, -0.0129413, 0.0127360
1: 0.9889134, 1.0106025, 0.9888791, 1.0119207, -0.0230073, 0.0217234
2: -0.0140034, 0.0059488, -0.0140627, 0.0066363, -0.0201364, 0.0195159
3: -0.0002013, 0.0059422, -0.0003871, 0.0059564, -0.0061576, 0.0063294
4: -0.0071524, 0.0094845, -0.0080262, 0.0095313, -0.0166838, 0.0175107
5: -0.0023368, 0.0110275, -0.0026392, 0.0110606, -0.0133974, 0.0136667
6: -0.0094139, 0.0036960, -0.0107637, 0.0037425, -0.0131564, 0.0144597
7: -0.0115525, 0.0002102, -0.0115763, 0.0006917, -0.0122442, 0.0117865
8: -0.0136829, 0.0166503, -0.0143559, 0.0167282, -0.0302266, 0.0308292
9: -0.0097618, 0.0076900, -0.0098050, 0.0080849, -0.0178466, 0.0174951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178403, upper bound: 0.0188474
time: 1.84 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179765, upper bound: 0.0188549
time: 2.14 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0063721, 0.0046130, -0.0060895, 0.0042431, -0.0106152, 0.0107025
1: 0.9889973, 1.0096678, 0.9892717, 1.0091066, -0.0201093, 0.0203961
2: -0.0138569, 0.0054612, -0.0133799, 0.0051686, -0.0185707, 0.0183464
3: -0.0000695, 0.0059073, 0.0000096, 0.0057936, -0.0058630, 0.0058977
4: -0.0065328, 0.0093687, -0.0061609, 0.0089917, -0.0155245, 0.0155296
5: -0.0021224, 0.0109456, -0.0019937, 0.0106792, -0.0128016, 0.0129394
6: -0.0084568, 0.0035812, -0.0078823, 0.0032072, -0.0116640, 0.0114635
7: -0.0114935, -0.0001313, -0.0113017, -0.0003363, -0.0111573, 0.0111703
8: -0.0132056, 0.0164579, -0.0129191, 0.0158312, -0.0288613, 0.0292150
9: -0.0096549, 0.0074100, -0.0093069, 0.0072420, -0.0168969, 0.0167169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175766, upper bound: 0.0178639
time: 1.74 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175678, upper bound: 0.0178034
time: 2.20 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0065253, 0.0048136, -0.0060705, 0.0042182, -0.0107435, 0.0108841
1: 0.9888097, 1.0099717, 0.9886777, 1.0090690, -0.0202593, 0.0212941
2: -0.0141835, 0.0056199, -0.0144133, 0.0051489, -0.0189069, 0.0194465
3: -0.0001124, 0.0059852, 0.0000149, 0.0060400, -0.0061523, 0.0059702
4: -0.0067345, 0.0096269, -0.0061359, 0.0098084, -0.0165429, 0.0157628
5: -0.0021922, 0.0111281, -0.0019851, 0.0112565, -0.0134487, 0.0131132
6: -0.0087683, 0.0038373, -0.0078437, 0.0040174, -0.0127857, 0.0116809
7: -0.0116250, -0.0000202, -0.0117174, -0.0003501, -0.0112749, 0.0116972
8: -0.0133609, 0.0168870, -0.0128998, 0.0171889, -0.0303580, 0.0296295
9: -0.0098932, 0.0075012, -0.0100608, 0.0072307, -0.0171239, 0.0175620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178274, upper bound: 0.0180812
time: 1.98 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175847, upper bound: 0.0180534
time: 2.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0054978, 0.0034684, -0.0073312, 0.0058686, -0.0113664, 0.0107996
1: 0.9885144, 1.0079315, 0.9888824, 1.0115720, -0.0230576, 0.0190490
2: -0.0146970, 0.0045557, -0.0140570, 0.0064545, -0.0206837, 0.0182017
3: 0.0001753, 0.0061076, -0.0003380, 0.0059550, -0.0057797, 0.0064456
4: -0.0053820, 0.0100327, -0.0077952, 0.0095268, -0.0149089, 0.0178279
5: -0.0017242, 0.0114151, -0.0025593, 0.0110574, -0.0127817, 0.0139743
6: -0.0066792, 0.0042399, -0.0104067, 0.0037380, -0.0104173, 0.0146466
7: -0.0118316, -0.0007655, -0.0115741, 0.0005644, -0.0123959, 0.0108085
8: -0.0123192, 0.0175617, -0.0141779, 0.0167207, -0.0288777, 0.0315813
9: -0.0102679, 0.0068900, -0.0098009, 0.0079804, -0.0182483, 0.0166909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178403, upper bound: 0.0179457
time: 1.89 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179763, upper bound: 0.0179805
time: 2.19 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0066576, 0.0049868, -0.0073312, 0.0058686, -0.0125262, 0.0123180
1: 0.9888048, 1.0102346, 0.9888824, 1.0115720, -0.0227672, 0.0213522
2: -0.0141923, 0.0057569, -0.0140570, 0.0064545, -0.0201907, 0.0193345
3: -0.0001494, 0.0059873, -0.0003380, 0.0059550, -0.0061044, 0.0063253
4: -0.0069086, 0.0096338, -0.0077952, 0.0095268, -0.0164354, 0.0174290
5: -0.0022525, 0.0111330, -0.0025593, 0.0110574, -0.0133099, 0.0136923
6: -0.0090373, 0.0038442, -0.0104067, 0.0037380, -0.0127754, 0.0142509
7: -0.0116285, 0.0000758, -0.0115741, 0.0005644, -0.0121929, 0.0116498
8: -0.0134950, 0.0168986, -0.0141779, 0.0167207, -0.0300405, 0.0309146
9: -0.0098996, 0.0075799, -0.0098009, 0.0079804, -0.0178801, 0.0173807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178403, upper bound: 0.0179427
time: 2.05 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179763, upper bound: 0.0188046
time: 2.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.85 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0179464, upper bound: 0.0181628
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0179464, upper bound: 0.0181630
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0178274, upper bound: 0.0181153
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0175848, upper bound: 0.0180893
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0178403, upper bound: 0.0180550
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0179765, upper bound: 0.0181392
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0178403, upper bound: 0.0188474
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0179765, upper bound: 0.0188549
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0175766, upper bound: 0.0178639
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0175678, upper bound: 0.0178034
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0178274, upper bound: 0.0180812
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0175847, upper bound: 0.0180534
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0178403, upper bound: 0.0179457
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0179763, upper bound: 0.0179805
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0178403, upper bound: 0.0179427
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 1, lower bound: -0.0179763, upper bound: 0.0188046

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0065870, 0.0048945, -0.0056188, 0.0036269, -0.0102139, 0.0105132
1: 0.9891065, 1.0100946, 0.9893032, 1.0081719, -0.0190654, 0.0207914
2: -0.0136676, 0.0056838, -0.0133249, 0.0046810, -0.0178351, 0.0184718
3: -0.0001297, 0.0058622, 0.0001414, 0.0057805, -0.0059101, 0.0057207
4: -0.0068157, 0.0092192, -0.0055413, 0.0089483, -0.0157640, 0.0147605
5: -0.0022203, 0.0108399, -0.0017793, 0.0106484, -0.0128687, 0.0126192
6: -0.0088939, 0.0034327, -0.0069253, 0.0031640, -0.0120579, 0.0103580
7: -0.0114174, 0.0000246, -0.0112795, -0.0006777, -0.0107397, 0.0113042
8: -0.0134235, 0.0162092, -0.0124419, 0.0157589, -0.0289967, 0.0284737
9: -0.0095168, 0.0075379, -0.0092668, 0.0069620, -0.0164788, 0.0168046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177947, upper bound: 0.0178442
time: 2.09 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175679, upper bound: 0.0178357
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0065870, 0.0048945, -0.0054014, 0.0033423, -0.0099294, 0.0102959
1: 0.9891065, 1.0100946, 0.9892157, 1.0077401, -0.0186337, 0.0208790
2: -0.0136676, 0.0056838, -0.0134773, 0.0044560, -0.0176521, 0.0186723
3: -0.0001297, 0.0058622, 0.0002023, 0.0058168, -0.0059464, 0.0056599
4: -0.0068157, 0.0092192, -0.0052552, 0.0090687, -0.0158844, 0.0144744
5: -0.0022203, 0.0108399, -0.0016803, 0.0107336, -0.0129539, 0.0125202
6: -0.0088939, 0.0034327, -0.0064834, 0.0032835, -0.0121774, 0.0099161
7: -0.0114174, 0.0000246, -0.0113408, -0.0008354, -0.0105820, 0.0113655
8: -0.0134235, 0.0162092, -0.0122215, 0.0159591, -0.0292105, 0.0282664
9: -0.0095168, 0.0075379, -0.0093779, 0.0068327, -0.0163495, 0.0169158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177947, upper bound: 0.0178443
time: 2.68 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175679, upper bound: 0.0178359
time: 1.76 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0067429, 0.0050986, -0.0060789, 0.0042293, -0.0109722, 0.0111775
1: 0.9889179, 1.0104041, 0.9888449, 1.0090857, -0.0201677, 0.0215592
2: -0.0139950, 0.0058453, -0.0141222, 0.0051576, -0.0186802, 0.0193597
3: -0.0001733, 0.0059402, 0.0000126, 0.0059706, -0.0061439, 0.0059277
4: -0.0070209, 0.0094779, -0.0061470, 0.0095785, -0.0165994, 0.0156249
5: -0.0022913, 0.0110228, -0.0019889, 0.0110939, -0.0133853, 0.0130117
6: -0.0092108, 0.0036894, -0.0078608, 0.0037892, -0.0130001, 0.0115503
7: -0.0115491, 0.0001377, -0.0116003, -0.0003440, -0.0112052, 0.0117380
8: -0.0135816, 0.0166393, -0.0129084, 0.0168065, -0.0301871, 0.0293786
9: -0.0097557, 0.0076306, -0.0098485, 0.0072357, -0.0169914, 0.0174791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178274, upper bound: 0.0181153
time: 1.68 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178274, upper bound: 0.0181153
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0066133, 0.0049289, -0.0065195, 0.0048060, -0.0114193, 0.0114483
1: 0.9890254, 1.0101466, 0.9889697, 1.0099603, -0.0209349, 0.0211769
2: -0.0138083, 0.0057110, -0.0139052, 0.0056139, -0.0189409, 0.0190329
3: -0.0001370, 0.0058957, -0.0001107, 0.0059188, -0.0060558, 0.0060065
4: -0.0068503, 0.0093303, -0.0067268, 0.0094069, -0.0162572, 0.0160571
5: -0.0022323, 0.0109184, -0.0021896, 0.0109726, -0.0132049, 0.0131080
6: -0.0089473, 0.0035430, -0.0087565, 0.0036190, -0.0125663, 0.0122996
7: -0.0114740, 0.0000436, -0.0115130, -0.0000244, -0.0114496, 0.0115566
8: -0.0134502, 0.0163940, -0.0133550, 0.0165213, -0.0297730, 0.0295807
9: -0.0096194, 0.0075535, -0.0096901, 0.0074977, -0.0171171, 0.0172436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178359
time: 2.56 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178899
time: 1.92 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0056029, 0.0036062, -0.0073470, 0.0058893, -0.0114923, 0.0109531
1: 0.9893032, 1.0081404, 0.9890676, 1.0116036, -0.0223004, 0.0190729
2: -0.0133249, 0.0046647, -0.0137353, 0.0064709, -0.0192846, 0.0179076
3: 0.0001458, 0.0057805, -0.0003424, 0.0058783, -0.0057325, 0.0061229
4: -0.0055205, 0.0089483, -0.0078160, 0.0092726, -0.0147931, 0.0167643
5: -0.0017721, 0.0106484, -0.0025665, 0.0108777, -0.0126498, 0.0132149
6: -0.0068931, 0.0031640, -0.0104389, 0.0034858, -0.0103789, 0.0136029
7: -0.0112795, -0.0006892, -0.0114446, 0.0005758, -0.0118554, 0.0107554
8: -0.0124258, 0.0157589, -0.0141940, 0.0162981, -0.0285460, 0.0297791
9: -0.0092668, 0.0069526, -0.0095662, 0.0079899, -0.0172566, 0.0165188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0181152, upper bound: 0.0180543
time: 2.31 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0181152, upper bound: 0.0180550
time: 2.08 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0055717, 0.0035652, -0.0075067, 0.0060985, -0.0116702, 0.0110720
1: 0.9887081, 1.0080786, 0.9888791, 1.0119207, -0.0232126, 0.0191995
2: -0.0143600, 0.0046323, -0.0140627, 0.0066363, -0.0203984, 0.0182271
3: 0.0001546, 0.0060273, -0.0003871, 0.0059564, -0.0058018, 0.0064144
4: -0.0054793, 0.0097664, -0.0080262, 0.0095313, -0.0150107, 0.0177926
5: -0.0017579, 0.0112267, -0.0026392, 0.0110606, -0.0128185, 0.0138660
6: -0.0068296, 0.0039757, -0.0107637, 0.0037425, -0.0105721, 0.0147394
7: -0.0116960, -0.0007119, -0.0115763, 0.0006917, -0.0123877, 0.0108644
8: -0.0123941, 0.0171190, -0.0143559, 0.0167282, -0.0289470, 0.0312857
9: -0.0100220, 0.0069340, -0.0098050, 0.0080849, -0.0181069, 0.0167390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184083, upper bound: 0.0181392
time: 2.32 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184083, upper bound: 0.0181385
time: 1.75 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0067018, 0.0050447, -0.0073470, 0.0058893, -0.0125911, 0.0123917
1: 0.9896147, 1.0103225, 0.9890676, 1.0116036, -0.0219889, 0.0212549
2: -0.0127832, 0.0058027, -0.0137353, 0.0064709, -0.0187767, 0.0189985
3: -0.0001618, 0.0056513, -0.0003424, 0.0058783, -0.0060401, 0.0059937
4: -0.0069668, 0.0085202, -0.0078160, 0.0092726, -0.0162394, 0.0163361
5: -0.0022726, 0.0103457, -0.0025665, 0.0108777, -0.0131502, 0.0129122
6: -0.0091272, 0.0027393, -0.0104389, 0.0034858, -0.0126130, 0.0131782
7: -0.0110616, 0.0001078, -0.0114446, 0.0005758, -0.0116375, 0.0115525
8: -0.0135399, 0.0150472, -0.0141940, 0.0162981, -0.0296548, 0.0290653
9: -0.0088715, 0.0076061, -0.0095662, 0.0079899, -0.0168614, 0.0171723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184674, upper bound: 0.0188474
time: 2.05 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184674, upper bound: 0.0188476
time: 2.42 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0067112, 0.0050570, -0.0075067, 0.0060985, -0.0128096, 0.0125637
1: 0.9889947, 1.0103409, 0.9888791, 1.0119207, -0.0229260, 0.0214618
2: -0.0138617, 0.0058124, -0.0140627, 0.0066363, -0.0199119, 0.0193813
3: -0.0001644, 0.0059084, -0.0003871, 0.0059564, -0.0061208, 0.0062956
4: -0.0069791, 0.0093725, -0.0080262, 0.0095313, -0.0165105, 0.0173987
5: -0.0022769, 0.0109483, -0.0026392, 0.0110606, -0.0133375, 0.0135875
6: -0.0091462, 0.0035849, -0.0107637, 0.0037425, -0.0128887, 0.0143486
7: -0.0114955, 0.0001146, -0.0115763, 0.0006917, -0.0121872, 0.0116910
8: -0.0135494, 0.0164641, -0.0143559, 0.0167282, -0.0300941, 0.0306223
9: -0.0096584, 0.0076117, -0.0098050, 0.0080849, -0.0177433, 0.0174167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0187726, upper bound: 0.0188549
time: 1.91 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0187726, upper bound: 0.0188549
time: 2.59 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0061729, 0.0043522, -0.0060895, 0.0042431, -0.0104160, 0.0104418
1: 0.9891566, 1.0092722, 0.9892717, 1.0091066, -0.0199500, 0.0200005
2: -0.0135801, 0.0052549, -0.0133799, 0.0051686, -0.0182908, 0.0181430
3: -0.0000137, 0.0058413, 0.0000096, 0.0057936, -0.0058073, 0.0058317
4: -0.0062706, 0.0091500, -0.0061609, 0.0089917, -0.0152623, 0.0153109
5: -0.0020317, 0.0107910, -0.0019937, 0.0106792, -0.0127108, 0.0127847
6: -0.0080518, 0.0033641, -0.0078823, 0.0032072, -0.0112590, 0.0112465
7: -0.0113822, -0.0002758, -0.0113017, -0.0003363, -0.0110459, 0.0110258
8: -0.0130036, 0.0160943, -0.0129191, 0.0158312, -0.0286600, 0.0288521
9: -0.0094530, 0.0072916, -0.0093069, 0.0072420, -0.0166950, 0.0165984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173149, upper bound: 0.0173252
time: 1.90 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177447
time: 2.19 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0065384, 0.0048308, -0.0059732, 0.0040908, -0.0106293, 0.0108040
1: 0.9893231, 1.0099981, 0.9893684, 1.0088755, -0.0195524, 0.0206296
2: -0.0132905, 0.0056335, -0.0132114, 0.0050481, -0.0179237, 0.0183505
3: -0.0001161, 0.0057722, 0.0000422, 0.0057534, -0.0058694, 0.0057301
4: -0.0067518, 0.0089211, -0.0060078, 0.0088585, -0.0156103, 0.0149288
5: -0.0021982, 0.0106291, -0.0019407, 0.0105849, -0.0127831, 0.0125699
6: -0.0087951, 0.0031370, -0.0076458, 0.0030750, -0.0118701, 0.0107828
7: -0.0112657, -0.0000107, -0.0112339, -0.0004207, -0.0108450, 0.0112232
8: -0.0133743, 0.0157136, -0.0128011, 0.0156097, -0.0288099, 0.0283629
9: -0.0092416, 0.0075090, -0.0091839, 0.0071728, -0.0164144, 0.0166929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172685, upper bound: 0.0171903
time: 2.37 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174404, upper bound: 0.0176854
time: 1.95 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0065253, 0.0048136, -0.0058890, 0.0039806, -0.0105059, 0.0107026
1: 0.9888097, 1.0099717, 0.9888488, 1.0087084, -0.0198987, 0.0211229
2: -0.0141835, 0.0056199, -0.0141155, 0.0049609, -0.0187219, 0.0191524
3: -0.0001124, 0.0059852, 0.0000658, 0.0059690, -0.0060813, 0.0059194
4: -0.0067345, 0.0096269, -0.0058970, 0.0095731, -0.0163076, 0.0155238
5: -0.0021922, 0.0111281, -0.0019024, 0.0110901, -0.0132823, 0.0130305
6: -0.0087683, 0.0038373, -0.0074746, 0.0037839, -0.0125522, 0.0113119
7: -0.0116250, -0.0000202, -0.0115976, -0.0004818, -0.0111432, 0.0115774
8: -0.0133609, 0.0168870, -0.0127158, 0.0167976, -0.0299678, 0.0294461
9: -0.0098932, 0.0075012, -0.0098436, 0.0071227, -0.0170159, 0.0173447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176709, upper bound: 0.0180780
time: 2.19 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176709, upper bound: 0.0178501
time: 1.91 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0063980, 0.0046469, -0.0063314, 0.0045598, -0.0109578, 0.0109784
1: 0.9889189, 1.0097189, 0.9889732, 1.0095868, -0.0206679, 0.0207457
2: -0.0139935, 0.0054880, -0.0138987, 0.0054191, -0.0189781, 0.0188270
3: -0.0000767, 0.0059399, -0.0000581, 0.0059173, -0.0059940, 0.0059980
4: -0.0065669, 0.0094767, -0.0064793, 0.0094018, -0.0159687, 0.0159560
5: -0.0021342, 0.0110220, -0.0021039, 0.0109690, -0.0131032, 0.0131259
6: -0.0085095, 0.0036883, -0.0083742, 0.0036140, -0.0121234, 0.0120625
7: -0.0115486, -0.0001125, -0.0115104, -0.0001608, -0.0113877, 0.0113979
8: -0.0132318, 0.0166374, -0.0131644, 0.0165128, -0.0295563, 0.0296437
9: -0.0097546, 0.0074254, -0.0096854, 0.0073859, -0.0171405, 0.0171109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178036
time: 2.59 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178424
time: 1.79 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0054014, 0.0033423, -0.0071733, 0.0056619, -0.0110633, 0.0105156
1: 0.9892157, 1.0077401, 0.9890708, 1.0112586, -0.0220429, 0.0186693
2: -0.0134773, 0.0044560, -0.0137295, 0.0062910, -0.0193143, 0.0177160
3: 0.0002023, 0.0058168, -0.0002938, 0.0058769, -0.0056747, 0.0061105
4: -0.0052552, 0.0090687, -0.0075873, 0.0092680, -0.0145233, 0.0166560
5: -0.0016803, 0.0107336, -0.0024873, 0.0108745, -0.0125548, 0.0132209
6: -0.0064834, 0.0032835, -0.0100857, 0.0034813, -0.0099647, 0.0133692
7: -0.0113408, -0.0008354, -0.0114423, 0.0004498, -0.0117907, 0.0106069
8: -0.0122215, 0.0159591, -0.0140178, 0.0162905, -0.0283457, 0.0298212
9: -0.0093779, 0.0068327, -0.0095619, 0.0078865, -0.0172645, 0.0163947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178130, upper bound: 0.0177946
time: 1.87 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178036, upper bound: 0.0175671
time: 1.90 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0053663, 0.0032963, -0.0073312, 0.0058686, -0.0112349, 0.0106275
1: 0.9885947, 1.0076706, 0.9888824, 1.0115720, -0.0229774, 0.0187882
2: -0.0145576, 0.0044196, -0.0140570, 0.0064545, -0.0204595, 0.0180661
3: 0.0002121, 0.0060744, -0.0003380, 0.0059550, -0.0057429, 0.0064124
4: -0.0052090, 0.0099225, -0.0077952, 0.0095268, -0.0147358, 0.0177177
5: -0.0016643, 0.0113371, -0.0025593, 0.0110574, -0.0127218, 0.0138964
6: -0.0064119, 0.0041306, -0.0104067, 0.0037380, -0.0101500, 0.0145373
7: -0.0117755, -0.0008609, -0.0115741, 0.0005644, -0.0123398, 0.0107131
8: -0.0121859, 0.0173785, -0.0141779, 0.0167207, -0.0287451, 0.0313772
9: -0.0101661, 0.0068118, -0.0098009, 0.0079804, -0.0181466, 0.0166127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180812, upper bound: 0.0178267
time: 2.01 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180541, upper bound: 0.0175847
time: 1.92 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0065028, 0.0047842, -0.0071733, 0.0056619, -0.0121647, 0.0119574
1: 0.9895087, 1.0099273, 0.9890708, 1.0112586, -0.0217499, 0.0208564
2: -0.0129676, 0.0055966, -0.0137295, 0.0062910, -0.0188242, 0.0188392
3: -0.0001061, 0.0056953, -0.0002938, 0.0058769, -0.0059830, 0.0059890
4: -0.0067048, 0.0086659, -0.0075873, 0.0092680, -0.0159729, 0.0162532
5: -0.0021819, 0.0104488, -0.0024873, 0.0108745, -0.0130564, 0.0129361
6: -0.0087226, 0.0028839, -0.0100857, 0.0034813, -0.0122038, 0.0129696
7: -0.0111358, -0.0000365, -0.0114423, 0.0004498, -0.0115856, 0.0114058
8: -0.0133381, 0.0152895, -0.0140178, 0.0162905, -0.0294537, 0.0291476
9: -0.0090061, 0.0074878, -0.0095619, 0.0078865, -0.0168926, 0.0170497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176273, upper bound: 0.0185455
time: 1.88 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0181953, upper bound: 0.0185320
time: 2.41 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0065247, 0.0048129, -0.0073312, 0.0058686, -0.0123933, 0.0121440
1: 0.9888845, 1.0099707, 0.9888824, 1.0115720, -0.0226875, 0.0210882
2: -0.0140534, 0.0056193, -0.0140570, 0.0064545, -0.0199693, 0.0191980
3: -0.0001122, 0.0059542, -0.0003380, 0.0059550, -0.0060672, 0.0062921
4: -0.0067337, 0.0095240, -0.0077952, 0.0095268, -0.0162605, 0.0173192
5: -0.0021919, 0.0110554, -0.0025593, 0.0110574, -0.0132494, 0.0136147
6: -0.0087671, 0.0037353, -0.0104067, 0.0037380, -0.0125052, 0.0141420
7: -0.0115726, -0.0000206, -0.0115741, 0.0005644, -0.0121370, 0.0115534
8: -0.0133603, 0.0167161, -0.0141779, 0.0167207, -0.0299067, 0.0307144
9: -0.0097983, 0.0075008, -0.0098009, 0.0079804, -0.0177787, 0.0173017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0185085, upper bound: 0.0185799
time: 2.11 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184999, upper bound: 0.0185326
time: 2.45 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.26 seconds
IS_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0177947, upper bound: 0.0178442
IS_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0175679, upper bound: 0.0178357
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0177947, upper bound: 0.0178443
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0175679, upper bound: 0.0178359
IS_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0178274, upper bound: 0.0181153
IS_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0178274, upper bound: 0.0181153
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178359
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178899
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0181152, upper bound: 0.0180543
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0181152, upper bound: 0.0180550
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0184083, upper bound: 0.0181392
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0184083, upper bound: 0.0181385
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0184674, upper bound: 0.0188474
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0184674, upper bound: 0.0188476
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0187726, upper bound: 0.0188549
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0187726, upper bound: 0.0188549
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0173149, upper bound: 0.0173252
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177447
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0172685, upper bound: 0.0171903
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0174404, upper bound: 0.0176854
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0176709, upper bound: 0.0180780
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0176709, upper bound: 0.0178501
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178036
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178424
IS_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0178130, upper bound: 0.0177946
IS_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0178036, upper bound: 0.0175671
IS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0180812, upper bound: 0.0178267
IS_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0180541, upper bound: 0.0175847
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0176273, upper bound: 0.0185455
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0181953, upper bound: 0.0185320
IS_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0185085, upper bound: 0.0185799
IS_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 1, lower bound: -0.0184999, upper bound: 0.0185326

## BFS IS instance: IS_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0065870, 0.0048945, -0.0054367, 0.0033886, -0.0099756, 0.0103312
1: 0.9891065, 1.0100946, 0.9894825, 1.0078105, -0.0187040, 0.0206121
2: -0.0136676, 0.0056838, -0.0130130, 0.0044925, -0.0176481, 0.0181656
3: -0.0001297, 0.0058622, 0.0001924, 0.0057061, -0.0058358, 0.0056698
4: -0.0068157, 0.0092192, -0.0053017, 0.0087018, -0.0155175, 0.0145208
5: -0.0022203, 0.0108399, -0.0016964, 0.0104742, -0.0126945, 0.0125363
6: -0.0088939, 0.0034327, -0.0065552, 0.0029195, -0.0118134, 0.0099879
7: -0.0114174, 0.0000246, -0.0111541, -0.0008098, -0.0106076, 0.0111787
8: -0.0134235, 0.0162092, -0.0122573, 0.0153492, -0.0285847, 0.0282896
9: -0.0095168, 0.0075379, -0.0090392, 0.0068537, -0.0163705, 0.0165771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175043, upper bound: 0.0172726
time: 2.38 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177565, upper bound: 0.0177439
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0064577, 0.0047252, -0.0059125, 0.0040114, -0.0104691, 0.0106376
1: 0.9892125, 1.0098376, 0.9895843, 1.0087552, -0.0195427, 0.0202533
2: -0.0134826, 0.0055499, -0.0128359, 0.0049852, -0.0179494, 0.0178718
3: -0.0000934, 0.0058181, 0.0000592, 0.0056638, -0.0057573, 0.0057589
4: -0.0066455, 0.0090729, -0.0059279, 0.0085618, -0.0152073, 0.0150008
5: -0.0021614, 0.0107365, -0.0019131, 0.0103752, -0.0125366, 0.0126496
6: -0.0086309, 0.0032877, -0.0075224, 0.0027805, -0.0114115, 0.0108101
7: -0.0113430, -0.0000692, -0.0110828, -0.0004647, -0.0108783, 0.0110136
8: -0.0132924, 0.0159661, -0.0127396, 0.0151163, -0.0282272, 0.0285269
9: -0.0093818, 0.0074610, -0.0089099, 0.0071367, -0.0165185, 0.0163709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169526, upper bound: 0.0174123
time: 1.91 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174902, upper bound: 0.0177373
time: 2.72 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0065870, 0.0048945, -0.0052226, 0.0031082, -0.0096952, 0.0101170
1: 0.9891065, 1.0100946, 0.9893860, 1.0073853, -0.0182788, 0.0207087
2: -0.0136676, 0.0056838, -0.0131811, 0.0042707, -0.0174692, 0.0183785
3: -0.0001297, 0.0058622, 0.0002523, 0.0057462, -0.0058758, 0.0056098
4: -0.0068157, 0.0092192, -0.0050199, 0.0088346, -0.0156503, 0.0142390
5: -0.0022203, 0.0108399, -0.0015989, 0.0105680, -0.0127884, 0.0124388
6: -0.0088939, 0.0034327, -0.0061198, 0.0030512, -0.0119451, 0.0095526
7: -0.0114174, 0.0000246, -0.0112217, -0.0009652, -0.0104523, 0.0112463
8: -0.0134235, 0.0162092, -0.0120402, 0.0155699, -0.0288220, 0.0280846
9: -0.0095168, 0.0075379, -0.0091618, 0.0067264, -0.0162432, 0.0166997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174202, upper bound: 0.0172410
time: 1.80 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176675, upper bound: 0.0177281
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0064577, 0.0047252, -0.0056850, 0.0037135, -0.0101712, 0.0104101
1: 0.9892125, 1.0098376, 0.9895132, 1.0083033, -0.0190908, 0.0203244
2: -0.0134826, 0.0055499, -0.0129599, 0.0047496, -0.0177621, 0.0180547
3: -0.0000934, 0.0058181, 0.0001229, 0.0056934, -0.0057869, 0.0056952
4: -0.0066455, 0.0090729, -0.0056284, 0.0086598, -0.0153053, 0.0147014
5: -0.0021614, 0.0107365, -0.0018095, 0.0104445, -0.0126059, 0.0125460
6: -0.0086309, 0.0032877, -0.0070599, 0.0028778, -0.0115087, 0.0103476
7: -0.0113430, -0.0000692, -0.0111327, -0.0006297, -0.0107133, 0.0110635
8: -0.0132924, 0.0159661, -0.0125090, 0.0152793, -0.0284036, 0.0283070
9: -0.0093818, 0.0074610, -0.0090004, 0.0070014, -0.0163832, 0.0164614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172687, upper bound: 0.0172138
time: 1.92 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174404, upper bound: 0.0177198
time: 2.13 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0067429, 0.0050986, -0.0054175, 0.0033633, -0.0101063, 0.0105160
1: 0.9889179, 1.0104041, 0.9888794, 1.0077721, -0.0188541, 0.0215247
2: -0.0139950, 0.0058453, -0.0140621, 0.0044726, -0.0179822, 0.0192862
3: -0.0001733, 0.0059402, 0.0001978, 0.0059563, -0.0061296, 0.0057425
4: -0.0070209, 0.0094779, -0.0052763, 0.0095309, -0.0165519, 0.0147542
5: -0.0022913, 0.0110228, -0.0016876, 0.0110603, -0.0133517, 0.0127104
6: -0.0092108, 0.0036894, -0.0065160, 0.0037421, -0.0129529, 0.0102055
7: -0.0115491, 0.0001377, -0.0115761, -0.0008238, -0.0107253, 0.0117138
8: -0.0135816, 0.0166393, -0.0122378, 0.0167275, -0.0301086, 0.0287025
9: -0.0097557, 0.0076306, -0.0098046, 0.0068423, -0.0165979, 0.0174352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174917, upper bound: 0.0175344
time: 2.23 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176989, upper bound: 0.0179970
time: 2.70 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0067429, 0.0050986, -0.0051739, 0.0030537, -0.0097966, 0.0102725
1: 0.9889179, 1.0104041, 0.9887618, 1.0072887, -0.0183707, 0.0216423
2: -0.0139950, 0.0058453, -0.0142668, 0.0042204, -0.0177730, 0.0195209
3: -0.0001733, 0.0059402, 0.0002659, 0.0060051, -0.0061784, 0.0056743
4: -0.0070209, 0.0094779, -0.0049558, 0.0096927, -0.0167136, 0.0144337
5: -0.0022913, 0.0110228, -0.0015767, 0.0111746, -0.0134660, 0.0125995
6: -0.0092108, 0.0036894, -0.0060209, 0.0039026, -0.0131134, 0.0097104
7: -0.0115491, 0.0001377, -0.0116585, -0.0010004, -0.0105487, 0.0117961
8: -0.0135816, 0.0166393, -0.0119909, 0.0169964, -0.0303836, 0.0284700
9: -0.0097557, 0.0076306, -0.0099540, 0.0066974, -0.0164531, 0.0175846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174917, upper bound: 0.0175344
time: 3.11 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176989, upper bound: 0.0179970
time: 1.98 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0064850, 0.0047609, -0.0065195, 0.0048060, -0.0112911, 0.0112804
1: 0.9897188, 1.0098919, 0.9889697, 1.0099603, -0.0202415, 0.0209221
2: -0.0126021, 0.0055782, -0.0139052, 0.0056139, -0.0177509, 0.0189483
3: -0.0001011, 0.0056081, -0.0001107, 0.0059188, -0.0060199, 0.0057189
4: -0.0066815, 0.0083770, -0.0067268, 0.0094069, -0.0160884, 0.0151038
5: -0.0021739, 0.0102445, -0.0021896, 0.0109726, -0.0131465, 0.0124341
6: -0.0086865, 0.0025973, -0.0087565, 0.0036190, -0.0123056, 0.0113538
7: -0.0109888, -0.0000494, -0.0115130, -0.0000244, -0.0109643, 0.0114636
8: -0.0133201, 0.0148092, -0.0133550, 0.0165213, -0.0296669, 0.0279926
9: -0.0087394, 0.0074772, -0.0096901, 0.0074977, -0.0162371, 0.0171674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0180865
time: 1.80 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0180865
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0064888, 0.0047659, -0.0065195, 0.0048060, -0.0112948, 0.0112854
1: 0.9891075, 1.0098995, 0.9889697, 1.0099603, -0.0208527, 0.0209298
2: -0.0136654, 0.0055822, -0.0139052, 0.0056139, -0.0187094, 0.0188772
3: -0.0001022, 0.0058617, -0.0001107, 0.0059188, -0.0060210, 0.0059724
4: -0.0066865, 0.0092174, -0.0067268, 0.0094069, -0.0160934, 0.0159442
5: -0.0021756, 0.0108387, -0.0021896, 0.0109726, -0.0131482, 0.0130282
6: -0.0086942, 0.0034310, -0.0087565, 0.0036190, -0.0123133, 0.0121876
7: -0.0114166, -0.0000466, -0.0115130, -0.0000244, -0.0113922, 0.0114664
8: -0.0133240, 0.0162063, -0.0133550, 0.0165213, -0.0296455, 0.0293696
9: -0.0095152, 0.0074795, -0.0096901, 0.0074977, -0.0170129, 0.0171696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178900
time: 2.44 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178898
time: 1.99 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0056029, 0.0036062, -0.0066859, 0.0050238, -0.0106268, 0.0102920
1: 0.9893032, 1.0081404, 0.9891016, 1.0102907, -0.0209875, 0.0190389
2: -0.0133249, 0.0046647, -0.0136760, 0.0057862, -0.0185898, 0.0178294
3: 0.0001458, 0.0057805, -0.0001573, 0.0058642, -0.0057183, 0.0059378
4: -0.0055205, 0.0089483, -0.0069458, 0.0092257, -0.0147462, 0.0158941
5: -0.0017721, 0.0106484, -0.0022653, 0.0108445, -0.0126166, 0.0129138
6: -0.0068931, 0.0031640, -0.0090948, 0.0034393, -0.0103324, 0.0122588
7: -0.0112795, -0.0006892, -0.0114208, 0.0000963, -0.0113758, 0.0107316
8: -0.0124258, 0.0157589, -0.0135237, 0.0162201, -0.0284685, 0.0291012
9: -0.0092668, 0.0069526, -0.0095229, 0.0075967, -0.0168634, 0.0164755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178132, upper bound: 0.0178630
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178039, upper bound: 0.0176093
time: 3.19 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0056029, 0.0036062, -0.0065038, 0.0047854, -0.0103884, 0.0101099
1: 0.9893032, 1.0081404, 0.9889925, 1.0099292, -0.0206259, 0.0191480
2: -0.0133249, 0.0046647, -0.0138658, 0.0055976, -0.0184409, 0.0180547
3: 0.0001458, 0.0057805, -0.0001064, 0.0059094, -0.0057636, 0.0058868
4: -0.0055205, 0.0089483, -0.0067061, 0.0093757, -0.0148962, 0.0156545
5: -0.0017721, 0.0106484, -0.0021824, 0.0109506, -0.0127227, 0.0128308
6: -0.0068931, 0.0031640, -0.0087246, 0.0035881, -0.0104812, 0.0118886
7: -0.0112795, -0.0006892, -0.0114971, -0.0000358, -0.0112437, 0.0108079
8: -0.0124258, 0.0157589, -0.0133391, 0.0164695, -0.0287249, 0.0289280
9: -0.0092668, 0.0069526, -0.0096614, 0.0074884, -0.0167551, 0.0166140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178132, upper bound: 0.0178623
time: 1.88 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178039, upper bound: 0.0176086
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0055717, 0.0035652, -0.0068428, 0.0052293, -0.0108010, 0.0104081
1: 0.9887081, 1.0080786, 0.9889134, 1.0106025, -0.0218943, 0.0191652
2: -0.0143600, 0.0046323, -0.0140034, 0.0059488, -0.0197009, 0.0181489
3: 0.0001546, 0.0060273, -0.0002013, 0.0059422, -0.0057877, 0.0062286
4: -0.0054793, 0.0097664, -0.0071524, 0.0094845, -0.0149638, 0.0169188
5: -0.0017579, 0.0112267, -0.0023368, 0.0110275, -0.0127854, 0.0135636
6: -0.0068296, 0.0039757, -0.0094139, 0.0036960, -0.0105256, 0.0133896
7: -0.0116960, -0.0007119, -0.0115525, 0.0002102, -0.0119062, 0.0108406
8: -0.0123941, 0.0171190, -0.0136829, 0.0166503, -0.0288696, 0.0306053
9: -0.0100220, 0.0069340, -0.0097618, 0.0076900, -0.0177121, 0.0166958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180836, upper bound: 0.0179253
time: 1.53 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180569, upper bound: 0.0176577
time: 2.08 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0055717, 0.0035652, -0.0066576, 0.0049868, -0.0105585, 0.0102228
1: 0.9887081, 1.0080786, 0.9888048, 1.0102346, -0.0215265, 0.0192738
2: -0.0143600, 0.0046323, -0.0141923, 0.0057569, -0.0195495, 0.0183754
3: 0.0001546, 0.0060273, -0.0001494, 0.0059873, -0.0058327, 0.0061767
4: -0.0054793, 0.0097664, -0.0069086, 0.0096338, -0.0151132, 0.0166750
5: -0.0017579, 0.0112267, -0.0022525, 0.0111330, -0.0128909, 0.0134792
6: -0.0068296, 0.0039757, -0.0090373, 0.0038442, -0.0106738, 0.0130130
7: -0.0116960, -0.0007119, -0.0116285, 0.0000758, -0.0117718, 0.0109166
8: -0.0123941, 0.0171190, -0.0134950, 0.0168986, -0.0291288, 0.0304293
9: -0.0100220, 0.0069340, -0.0098996, 0.0075799, -0.0176019, 0.0168336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180836, upper bound: 0.0179253
time: 1.69 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180569, upper bound: 0.0176577
time: 2.18 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0067018, 0.0050447, -0.0066859, 0.0050238, -0.0117256, 0.0117306
1: 0.9896147, 1.0103225, 0.9891016, 1.0102907, -0.0206760, 0.0212209
2: -0.0127832, 0.0058027, -0.0136760, 0.0057862, -0.0180799, 0.0189221
3: -0.0001618, 0.0056513, -0.0001573, 0.0058642, -0.0060260, 0.0058086
4: -0.0069668, 0.0085202, -0.0069458, 0.0092257, -0.0161925, 0.0154660
5: -0.0022726, 0.0103457, -0.0022653, 0.0108445, -0.0131171, 0.0126111
6: -0.0091272, 0.0027393, -0.0090948, 0.0034393, -0.0125665, 0.0118341
7: -0.0110616, 0.0001078, -0.0114208, 0.0000963, -0.0111579, 0.0115286
8: -0.0135399, 0.0150472, -0.0135237, 0.0162201, -0.0295772, 0.0283879
9: -0.0088715, 0.0076061, -0.0095229, 0.0075967, -0.0164682, 0.0171290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0182377, upper bound: 0.0185660
time: 3.24 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0181953, upper bound: 0.0185564
time: 2.29 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0067018, 0.0050447, -0.0065038, 0.0047854, -0.0114872, 0.0115485
1: 0.9896147, 1.0103225, 0.9889925, 1.0099292, -0.0203145, 0.0213300
2: -0.0127832, 0.0058027, -0.0138658, 0.0055976, -0.0179271, 0.0191466
3: -0.0001618, 0.0056513, -0.0001064, 0.0059094, -0.0060712, 0.0057576
4: -0.0069668, 0.0085202, -0.0067061, 0.0093757, -0.0163425, 0.0152263
5: -0.0022726, 0.0103457, -0.0021824, 0.0109506, -0.0132232, 0.0125281
6: -0.0091272, 0.0027393, -0.0087246, 0.0035881, -0.0127153, 0.0114639
7: -0.0110616, 0.0001078, -0.0114971, -0.0000358, -0.0110258, 0.0116050
8: -0.0135399, 0.0150472, -0.0133391, 0.0164695, -0.0298334, 0.0282138
9: -0.0088715, 0.0076061, -0.0096614, 0.0074884, -0.0163599, 0.0172675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0182377, upper bound: 0.0185658
time: 2.11 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0181953, upper bound: 0.0185565
time: 2.28 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0067112, 0.0050570, -0.0068428, 0.0052293, -0.0119405, 0.0118998
1: 0.9889947, 1.0103409, 0.9889134, 1.0106025, -0.0216078, 0.0214275
2: -0.0138617, 0.0058124, -0.0140034, 0.0059488, -0.0192128, 0.0193014
3: -0.0001644, 0.0059084, -0.0002013, 0.0059422, -0.0061066, 0.0061097
4: -0.0069791, 0.0093725, -0.0071524, 0.0094845, -0.0164636, 0.0165249
5: -0.0022769, 0.0109483, -0.0023368, 0.0110275, -0.0133044, 0.0132851
6: -0.0091462, 0.0035849, -0.0094139, 0.0036960, -0.0128422, 0.0129988
7: -0.0114955, 0.0001146, -0.0115525, 0.0002102, -0.0117056, 0.0116671
8: -0.0135494, 0.0164641, -0.0136829, 0.0166503, -0.0300166, 0.0299425
9: -0.0096584, 0.0076117, -0.0097618, 0.0076900, -0.0173484, 0.0173735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0187687, upper bound: 0.0185423
time: 2.47 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0187687, upper bound: 0.0185998
time: 2.38 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0067112, 0.0050570, -0.0066576, 0.0049868, -0.0116980, 0.0117145
1: 0.9889947, 1.0103409, 0.9888048, 1.0102346, -0.0212399, 0.0215361
2: -0.0138617, 0.0058124, -0.0141923, 0.0057569, -0.0190584, 0.0195298
3: -0.0001644, 0.0059084, -0.0001494, 0.0059873, -0.0061517, 0.0060579
4: -0.0069791, 0.0093725, -0.0069086, 0.0096338, -0.0166130, 0.0162811
5: -0.0022769, 0.0109483, -0.0022525, 0.0111330, -0.0134099, 0.0132008
6: -0.0091462, 0.0035849, -0.0090373, 0.0038442, -0.0129904, 0.0126222
7: -0.0114955, 0.0001146, -0.0116285, 0.0000758, -0.0115713, 0.0117431
8: -0.0135494, 0.0164641, -0.0134950, 0.0168986, -0.0302755, 0.0297656
9: -0.0096584, 0.0076117, -0.0098996, 0.0075799, -0.0172383, 0.0175114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0187687, upper bound: 0.0185423
time: 2.43 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0187687, upper bound: 0.0185998
time: 2.26 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0052515, 0.0031461, -0.0057938, 0.0038560, -0.0091075, 0.0089398
1: 0.9890762, 1.0074426, 0.9892812, 1.0085193, -0.0194431, 0.0181614
2: -0.0137200, 0.0043007, -0.0133632, 0.0048623, -0.0181084, 0.0171448
3: 0.0002442, 0.0058747, 0.0000924, 0.0057896, -0.0055454, 0.0057823
4: -0.0050579, 0.0092605, -0.0057716, 0.0089785, -0.0140364, 0.0150322
5: -0.0016121, 0.0108691, -0.0018590, 0.0106698, -0.0122818, 0.0127282
6: -0.0061786, 0.0034738, -0.0072811, 0.0031940, -0.0093726, 0.0107549
7: -0.0114385, -0.0009442, -0.0112949, -0.0005508, -0.0108877, 0.0103508
8: -0.0120695, 0.0162779, -0.0126193, 0.0158092, -0.0277068, 0.0287385
9: -0.0095550, 0.0067436, -0.0092947, 0.0070661, -0.0166211, 0.0160382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173152, upper bound: 0.0173252
time: 1.79 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173152, upper bound: 0.0173252
time: 2.01 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0058134, 0.0038817, -0.0060807, 0.0042316, -0.0100450, 0.0099624
1: 0.9891667, 1.0085585, 0.9892719, 1.0090892, -0.0199226, 0.0192866
2: -0.0135627, 0.0048827, -0.0133795, 0.0051595, -0.0182562, 0.0177110
3: 0.0000869, 0.0058372, 0.0000121, 0.0057935, -0.0057066, 0.0058251
4: -0.0057975, 0.0091362, -0.0061493, 0.0089915, -0.0147890, 0.0152855
5: -0.0018680, 0.0107813, -0.0019897, 0.0106789, -0.0125469, 0.0127710
6: -0.0073211, 0.0033505, -0.0078644, 0.0032069, -0.0105279, 0.0112149
7: -0.0113752, -0.0005365, -0.0113015, -0.0003427, -0.0110325, 0.0107650
8: -0.0126392, 0.0160713, -0.0129102, 0.0158307, -0.0282848, 0.0288199
9: -0.0094402, 0.0070778, -0.0093066, 0.0072367, -0.0166770, 0.0163844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177447
time: 2.51 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177447
time: 1.82 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0056803, 0.0037075, -0.0056772, 0.0037034, -0.0093838, 0.0093847
1: 0.9892157, 1.0082941, 0.9893781, 1.0082880, -0.0190723, 0.0189160
2: -0.0134774, 0.0047448, -0.0131947, 0.0047416, -0.0177702, 0.0174076
3: 0.0001242, 0.0058168, 0.0001250, 0.0057494, -0.0056252, 0.0056918
4: -0.0056223, 0.0090688, -0.0056183, 0.0088454, -0.0144677, 0.0146870
5: -0.0018074, 0.0107336, -0.0018060, 0.0105756, -0.0123830, 0.0125395
6: -0.0070505, 0.0032836, -0.0070442, 0.0030619, -0.0101124, 0.0103278
7: -0.0113409, -0.0006331, -0.0112272, -0.0006353, -0.0107055, 0.0105941
8: -0.0125043, 0.0159592, -0.0125012, 0.0155878, -0.0279174, 0.0283047
9: -0.0093780, 0.0069986, -0.0091717, 0.0069968, -0.0163748, 0.0161704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172687, upper bound: 0.0171903
time: 1.68 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172687, upper bound: 0.0171903
time: 2.03 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0061782, 0.0043592, -0.0059643, 0.0040793, -0.0102575, 0.0103236
1: 0.9893335, 1.0092827, 0.9893687, 1.0088580, -0.0195245, 0.0199140
2: -0.0132724, 0.0052604, -0.0132110, 0.0050389, -0.0178888, 0.0179210
3: -0.0000152, 0.0057680, 0.0000447, 0.0057533, -0.0057685, 0.0057233
4: -0.0062776, 0.0089068, -0.0059961, 0.0088583, -0.0151359, 0.0149030
5: -0.0020341, 0.0106191, -0.0019367, 0.0105848, -0.0126189, 0.0125558
6: -0.0080626, 0.0031229, -0.0076279, 0.0030747, -0.0111374, 0.0107507
7: -0.0112584, -0.0002720, -0.0112337, -0.0004271, -0.0108313, 0.0109618
8: -0.0130090, 0.0156900, -0.0127922, 0.0156093, -0.0284348, 0.0283303
9: -0.0092285, 0.0072947, -0.0091837, 0.0071675, -0.0163960, 0.0164784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174404, upper bound: 0.0176856
time: 1.88 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174404, upper bound: 0.0176854
time: 2.24 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0063891, 0.0046354, -0.0058890, 0.0039806, -0.0103697, 0.0105243
1: 0.9895137, 1.0097016, 0.9888488, 1.0087084, -0.0191947, 0.0208528
2: -0.0129588, 0.0054789, -0.0141155, 0.0049609, -0.0175112, 0.0190828
3: -0.0000743, 0.0056932, 0.0000658, 0.0059690, -0.0060432, 0.0056274
4: -0.0065553, 0.0086590, -0.0058970, 0.0095731, -0.0161284, 0.0145559
5: -0.0021302, 0.0104439, -0.0019024, 0.0110901, -0.0132203, 0.0123463
6: -0.0084915, 0.0028770, -0.0074746, 0.0037839, -0.0122754, 0.0103516
7: -0.0111322, -0.0001189, -0.0115976, -0.0004818, -0.0106505, 0.0114787
8: -0.0132229, 0.0152779, -0.0127158, 0.0167976, -0.0298543, 0.0278380
9: -0.0089997, 0.0074202, -0.0098436, 0.0071227, -0.0161224, 0.0172637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172412, upper bound: 0.0174474
time: 2.48 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175428, upper bound: 0.0179587
time: 2.36 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0063931, 0.0046406, -0.0058890, 0.0039806, -0.0103737, 0.0105295
1: 0.9888895, 1.0097095, 0.9888488, 1.0087084, -0.0198188, 0.0208607
2: -0.0140447, 0.0054830, -0.0141155, 0.0049609, -0.0184823, 0.0190162
3: -0.0000754, 0.0059521, 0.0000658, 0.0059690, -0.0060443, 0.0058863
4: -0.0065605, 0.0095171, -0.0058970, 0.0095731, -0.0161336, 0.0154141
5: -0.0021320, 0.0110505, -0.0019024, 0.0110901, -0.0132221, 0.0129529
6: -0.0084996, 0.0037284, -0.0074746, 0.0037839, -0.0122835, 0.0112030
7: -0.0115691, -0.0001161, -0.0115976, -0.0004818, -0.0110874, 0.0114815
8: -0.0132269, 0.0167046, -0.0127158, 0.0167976, -0.0298344, 0.0292414
9: -0.0097919, 0.0074226, -0.0098436, 0.0071227, -0.0169146, 0.0172661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172412, upper bound: 0.0173577
time: 1.93 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175428, upper bound: 0.0177302
time: 1.85 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0062654, 0.0044734, -0.0063314, 0.0045598, -0.0108253, 0.0108048
1: 0.9896188, 1.0094557, 0.9889732, 1.0095868, -0.0199680, 0.0204825
2: -0.0127761, 0.0053508, -0.0138987, 0.0054191, -0.0177720, 0.0187394
3: -0.0000396, 0.0056496, -0.0000581, 0.0059173, -0.0059569, 0.0057077
4: -0.0063924, 0.0085145, -0.0064793, 0.0094018, -0.0157942, 0.0149938
5: -0.0020739, 0.0103417, -0.0021039, 0.0109690, -0.0130429, 0.0124456
6: -0.0082400, 0.0027337, -0.0083742, 0.0036140, -0.0118540, 0.0111079
7: -0.0110587, -0.0002087, -0.0115104, -0.0001608, -0.0108979, 0.0113017
8: -0.0130975, 0.0150378, -0.0131644, 0.0165128, -0.0294444, 0.0280443
9: -0.0088663, 0.0073466, -0.0096854, 0.0073859, -0.0162522, 0.0170320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170998, upper bound: 0.0174030
time: 1.63 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173454, upper bound: 0.0179312
time: 1.99 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0062667, 0.0044751, -0.0063314, 0.0045598, -0.0108265, 0.0108065
1: 0.9889988, 1.0094585, 0.9889732, 1.0095868, -0.0205880, 0.0204853
2: -0.0138546, 0.0053521, -0.0138987, 0.0054191, -0.0187468, 0.0186918
3: -0.0000400, 0.0059068, -0.0000581, 0.0059173, -0.0059573, 0.0059649
4: -0.0063941, 0.0093669, -0.0064793, 0.0094018, -0.0157959, 0.0158462
5: -0.0020744, 0.0109444, -0.0021039, 0.0109690, -0.0130435, 0.0130483
6: -0.0082425, 0.0035794, -0.0083742, 0.0036140, -0.0118565, 0.0119535
7: -0.0114926, -0.0002078, -0.0115104, -0.0001608, -0.0113318, 0.0113026
8: -0.0130987, 0.0164549, -0.0131644, 0.0165128, -0.0294238, 0.0294387
9: -0.0096532, 0.0073474, -0.0096854, 0.0073859, -0.0170391, 0.0170328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170998, upper bound: 0.0173348
time: 1.92 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173454, upper bound: 0.0177209
time: 1.89 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0052226, 0.0031082, -0.0071733, 0.0056619, -0.0108845, 0.0102815
1: 0.9893860, 1.0073853, 0.9890708, 1.0112586, -0.0218726, 0.0183144
2: -0.0131811, 0.0042707, -0.0137295, 0.0062910, -0.0190205, 0.0175330
3: 0.0002523, 0.0057462, -0.0002938, 0.0058769, -0.0056246, 0.0060399
4: -0.0050199, 0.0088346, -0.0075873, 0.0092680, -0.0142879, 0.0164219
5: -0.0015989, 0.0105680, -0.0024873, 0.0108745, -0.0124733, 0.0130554
6: -0.0061198, 0.0030512, -0.0100857, 0.0034813, -0.0096011, 0.0131370
7: -0.0112217, -0.0009652, -0.0114423, 0.0004498, -0.0116715, 0.0104772
8: -0.0120402, 0.0155699, -0.0140178, 0.0162905, -0.0281640, 0.0294327
9: -0.0091618, 0.0067264, -0.0095619, 0.0078865, -0.0170484, 0.0162883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172160, upper bound: 0.0174217
time: 2.80 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176958, upper bound: 0.0176675
time: 2.31 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0056850, 0.0037135, -0.0070455, 0.0054947, -0.0111797, 0.0107591
1: 0.9895132, 1.0083033, 0.9891769, 1.0110049, -0.0214917, 0.0191264
2: -0.0129599, 0.0047496, -0.0135449, 0.0061587, -0.0186986, 0.0178307
3: 0.0001229, 0.0056934, -0.0002580, 0.0058329, -0.0057100, 0.0059514
4: -0.0056284, 0.0086598, -0.0074192, 0.0091221, -0.0147505, 0.0160790
5: -0.0018095, 0.0104445, -0.0024291, 0.0107713, -0.0125808, 0.0128736
6: -0.0070599, 0.0028778, -0.0098260, 0.0033365, -0.0103964, 0.0127038
7: -0.0111327, -0.0006297, -0.0113680, 0.0003572, -0.0114899, 0.0107383
8: -0.0125090, 0.0152793, -0.0138883, 0.0160479, -0.0283884, 0.0290159
9: -0.0090004, 0.0070014, -0.0094272, 0.0078106, -0.0168110, 0.0164286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0171903, upper bound: 0.0172685
time: 1.86 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176856, upper bound: 0.0174398
time: 1.91 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0051739, 0.0030537, -0.0073312, 0.0058686, -0.0110426, 0.0103848
1: 0.9887618, 1.0072887, 0.9888824, 1.0115720, -0.0228102, 0.0184063
2: -0.0142668, 0.0042204, -0.0140570, 0.0064545, -0.0201683, 0.0178405
3: 0.0002659, 0.0060051, -0.0003380, 0.0059550, -0.0056891, 0.0063430
4: -0.0049558, 0.0096927, -0.0077952, 0.0095268, -0.0144827, 0.0174878
5: -0.0015767, 0.0111746, -0.0025593, 0.0110574, -0.0126342, 0.0137339
6: -0.0060209, 0.0039026, -0.0104067, 0.0037380, -0.0097589, 0.0143093
7: -0.0116585, -0.0010004, -0.0115741, 0.0005644, -0.0122228, 0.0105736
8: -0.0119909, 0.0169964, -0.0141779, 0.0167207, -0.0285488, 0.0309969
9: -0.0099540, 0.0066974, -0.0098009, 0.0079804, -0.0179344, 0.0164983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175017, upper bound: 0.0174921
time: 2.58 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179626, upper bound: 0.0176988
time: 1.95 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0056034, 0.0036068, -0.0072031, 0.0057010, -0.0113045, 0.0108099
1: 0.9889076, 1.0081414, 0.9889897, 1.0113180, -0.0224103, 0.0191517
2: -0.0140132, 0.0046652, -0.0138704, 0.0063219, -0.0198159, 0.0180931
3: 0.0001457, 0.0059446, -0.0003021, 0.0059105, -0.0057649, 0.0062467
4: -0.0055211, 0.0094923, -0.0076267, 0.0093795, -0.0149006, 0.0171189
5: -0.0017723, 0.0110330, -0.0025009, 0.0109532, -0.0127256, 0.0135339
6: -0.0068941, 0.0037037, -0.0101465, 0.0035918, -0.0104859, 0.0138502
7: -0.0115564, -0.0006889, -0.0114990, 0.0004715, -0.0120280, 0.0108102
8: -0.0124263, 0.0166633, -0.0140481, 0.0164757, -0.0287364, 0.0305374
9: -0.0097690, 0.0069529, -0.0096648, 0.0079043, -0.0176733, 0.0166177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174592, upper bound: 0.0173286
time: 1.97 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179346, upper bound: 0.0174581
time: 1.84 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0065028, 0.0047842, -0.0069808, 0.0054100, -0.0119128, 0.0117650
1: 0.9895087, 1.0099273, 0.9892355, 1.0108763, -0.0213676, 0.0206918
2: -0.0129676, 0.0055966, -0.0134431, 0.0060916, -0.0185943, 0.0185504
3: -0.0001061, 0.0056953, -0.0002399, 0.0058087, -0.0059147, 0.0059352
4: -0.0067048, 0.0086659, -0.0073340, 0.0090417, -0.0157465, 0.0159999
5: -0.0021819, 0.0104488, -0.0023997, 0.0107145, -0.0128964, 0.0128484
6: -0.0087226, 0.0028839, -0.0096944, 0.0032567, -0.0119793, 0.0125783
7: -0.0111358, -0.0000365, -0.0113271, 0.0003102, -0.0114460, 0.0112906
8: -0.0133381, 0.0152895, -0.0138227, 0.0159142, -0.0290761, 0.0289515
9: -0.0090061, 0.0074878, -0.0093530, 0.0077721, -0.0167782, 0.0168408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178103, upper bound: 0.0182257
time: 2.40 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0181155, upper bound: 0.0184367
time: 2.24 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0063804, 0.0046240, -0.0073486, 0.0058915, -0.0122719, 0.0119726
1: 0.9896137, 1.0096842, 0.9893765, 1.0116066, -0.0219929, 0.0203077
2: -0.0127850, 0.0054698, -0.0131975, 0.0064725, -0.0187862, 0.0181916
3: -0.0000718, 0.0056517, -0.0003429, 0.0057501, -0.0058219, 0.0059946
4: -0.0065438, 0.0085215, -0.0078181, 0.0088476, -0.0153914, 0.0163396
5: -0.0021262, 0.0103467, -0.0025672, 0.0105772, -0.0127035, 0.0129139
6: -0.0084738, 0.0027407, -0.0104422, 0.0030642, -0.0115379, 0.0131829
7: -0.0110623, -0.0001253, -0.0112283, 0.0005770, -0.0116393, 0.0111030
8: -0.0132140, 0.0150495, -0.0141956, 0.0155916, -0.0286329, 0.0290798
9: -0.0088728, 0.0074150, -0.0091738, 0.0079908, -0.0168636, 0.0165888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176864, upper bound: 0.0181917
time: 2.18 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180748, upper bound: 0.0184205
time: 2.83 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0063280, 0.0045554, -0.0073312, 0.0058686, -0.0121966, 0.0118866
1: 0.9890420, 1.0095803, 0.9888824, 1.0115720, -0.0225300, 0.0206978
2: -0.0137796, 0.0054156, -0.0140570, 0.0064545, -0.0196898, 0.0189971
3: -0.0000571, 0.0058889, -0.0003380, 0.0059550, -0.0060122, 0.0062269
4: -0.0064748, 0.0093076, -0.0077952, 0.0095268, -0.0160017, 0.0171028
5: -0.0021024, 0.0109025, -0.0025593, 0.0110574, -0.0131598, 0.0134617
6: -0.0083673, 0.0035206, -0.0104067, 0.0037380, -0.0121053, 0.0139273
7: -0.0114625, -0.0001633, -0.0115741, 0.0005644, -0.0120268, 0.0114108
8: -0.0131609, 0.0163563, -0.0141779, 0.0167207, -0.0297074, 0.0303550
9: -0.0095985, 0.0073838, -0.0098009, 0.0079804, -0.0175790, 0.0171847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0185082, upper bound: 0.0182540
time: 2.49 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0185082, upper bound: 0.0182941
time: 1.91 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0066952, 0.0050360, -0.0072031, 0.0057010, -0.0123962, 0.0122391
1: 0.9892124, 1.0103091, 0.9889897, 1.0113180, -0.0221056, 0.0213194
2: -0.0134832, 0.0057958, -0.0138704, 0.0063219, -0.0193034, 0.0191621
3: -0.0001599, 0.0058182, -0.0003021, 0.0059105, -0.0060705, 0.0061203
4: -0.0069580, 0.0090734, -0.0076267, 0.0093795, -0.0163375, 0.0167000
5: -0.0022696, 0.0107368, -0.0025009, 0.0109532, -0.0132228, 0.0132378
6: -0.0091137, 0.0032881, -0.0101465, 0.0035918, -0.0127055, 0.0134346
7: -0.0113432, 0.0001030, -0.0114990, 0.0004715, -0.0118147, 0.0116021
8: -0.0135331, 0.0159668, -0.0140481, 0.0164757, -0.0298306, 0.0298363
9: -0.0093822, 0.0076022, -0.0096648, 0.0079043, -0.0172865, 0.0172670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184994, upper bound: 0.0182114
time: 2.49 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184994, upper bound: 0.0182487
time: 2.23 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.49 seconds
IS_A1_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0175043, upper bound: 0.0172726
IS_A1_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0177565, upper bound: 0.0177439
IS_A1_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0169526, upper bound: 0.0174123
IS_A1_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174902, upper bound: 0.0177373
IS_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174202, upper bound: 0.0172410
IS_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0176675, upper bound: 0.0177281
IS_A1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0172687, upper bound: 0.0172138
IS_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174404, upper bound: 0.0177198
IS_A1_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174917, upper bound: 0.0175344
IS_A1_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0176989, upper bound: 0.0179970
IS_A1_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174917, upper bound: 0.0175344
IS_A1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0176989, upper bound: 0.0179970
IS_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0180865
IS_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0180865
IS_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178900
IS_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174725, upper bound: 0.0178898
IS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0178132, upper bound: 0.0178630
IS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0178039, upper bound: 0.0176093
IS_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0178132, upper bound: 0.0178623
IS_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0178039, upper bound: 0.0176086
IS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0180836, upper bound: 0.0179253
IS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0180569, upper bound: 0.0176577
IS_A1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0180836, upper bound: 0.0179253
IS_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0180569, upper bound: 0.0176577
IS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0182377, upper bound: 0.0185660
IS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0181953, upper bound: 0.0185564
IS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0182377, upper bound: 0.0185658
IS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0181953, upper bound: 0.0185565
IS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0187687, upper bound: 0.0185423
IS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0187687, upper bound: 0.0185998
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0187687, upper bound: 0.0185423
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0187687, upper bound: 0.0185998
IS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0173152, upper bound: 0.0173252
IS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0173152, upper bound: 0.0173252
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177447
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177447
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0172687, upper bound: 0.0171903
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0172687, upper bound: 0.0171903
IS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174404, upper bound: 0.0176856
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174404, upper bound: 0.0176854
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0172412, upper bound: 0.0174474
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0175428, upper bound: 0.0179587
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0172412, upper bound: 0.0173577
IS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0175428, upper bound: 0.0177302
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0170998, upper bound: 0.0174030
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0173454, upper bound: 0.0179312
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0170998, upper bound: 0.0173348
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0173454, upper bound: 0.0177209
IS_A2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0172160, upper bound: 0.0174217
IS_A2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0176958, upper bound: 0.0176675
IS_A2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0171903, upper bound: 0.0172685
IS_A2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0176856, upper bound: 0.0174398
IS_A2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0175017, upper bound: 0.0174921
IS_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0179626, upper bound: 0.0176988
IS_A2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0174592, upper bound: 0.0173286
IS_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0179346, upper bound: 0.0174581
IS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0178103, upper bound: 0.0182257
IS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0181155, upper bound: 0.0184367
IS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0176864, upper bound: 0.0181917
IS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0180748, upper bound: 0.0184205
IS_A2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0185082, upper bound: 0.0182540
IS_A2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0185082, upper bound: 0.0182941
IS_A2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0184994, upper bound: 0.0182114
IS_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0184994, upper bound: 0.0182487

## BFS IS instance: IS_A1_B1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0056640, 0.0036861, -0.0051506, 0.0030140, -0.0086781, 0.0088368
1: 0.9890331, 1.0082619, 0.9894919, 1.0072423, -0.0182092, 0.0187700
2: -0.0137951, 0.0047280, -0.0129969, 0.0041963, -0.0174638, 0.0171744
3: 0.0001287, 0.0058926, 0.0002725, 0.0057022, -0.0055735, 0.0056201
4: -0.0056009, 0.0093198, -0.0049252, 0.0086890, -0.0142899, 0.0142450
5: -0.0018000, 0.0109110, -0.0015661, 0.0104651, -0.0122651, 0.0124772
6: -0.0070174, 0.0035326, -0.0059735, 0.0029068, -0.0099242, 0.0095062
7: -0.0114687, -0.0006449, -0.0111476, -0.0010173, -0.0104514, 0.0105026
8: -0.0124878, 0.0163766, -0.0119673, 0.0153279, -0.0276298, 0.0281652
9: -0.0096098, 0.0069889, -0.0090274, 0.0066836, -0.0162933, 0.0160164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175043, upper bound: 0.0172725
time: 2.35 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175043, upper bound: 0.0172726
time: 2.31 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0062249, 0.0044203, -0.0054278, 0.0033768, -0.0096017, 0.0098481
1: 0.9891161, 1.0093753, 0.9894828, 1.0077927, -0.0186766, 0.0198926
2: -0.0136506, 0.0053088, -0.0130127, 0.0044832, -0.0176138, 0.0177372
3: -0.0000283, 0.0058581, 0.0001949, 0.0057060, -0.0057343, 0.0056632
4: -0.0063390, 0.0092057, -0.0052899, 0.0087016, -0.0150406, 0.0144956
5: -0.0020554, 0.0108303, -0.0016923, 0.0104740, -0.0125293, 0.0125227
6: -0.0081575, 0.0034194, -0.0065370, 0.0029192, -0.0110767, 0.0099564
7: -0.0114106, -0.0002381, -0.0111539, -0.0008163, -0.0105943, 0.0109158
8: -0.0130563, 0.0161868, -0.0122482, 0.0153487, -0.0282090, 0.0282579
9: -0.0095044, 0.0073225, -0.0090390, 0.0068484, -0.0163528, 0.0163615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177565, upper bound: 0.0177438
time: 1.97 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177565, upper bound: 0.0177438
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0061605, 0.0043360, -0.0050530, 0.0028862, -0.0090466, 0.0093890
1: 0.9892223, 1.0092474, 0.9894814, 1.0070484, -0.0178260, 0.0197660
2: -0.0134657, 0.0052421, -0.0130153, 0.0040951, -0.0170446, 0.0177036
3: -0.0000102, 0.0058140, 0.0002998, 0.0057066, -0.0057169, 0.0055142
4: -0.0062543, 0.0090596, -0.0047966, 0.0087036, -0.0149579, 0.0138562
5: -0.0020260, 0.0107271, -0.0015216, 0.0104754, -0.0125015, 0.0122487
6: -0.0080266, 0.0032744, -0.0057750, 0.0029213, -0.0109479, 0.0090494
7: -0.0113362, -0.0002848, -0.0111550, -0.0010881, -0.0102480, 0.0108702
8: -0.0129910, 0.0159439, -0.0118683, 0.0153521, -0.0281593, 0.0276333
9: -0.0093695, 0.0072842, -0.0090409, 0.0066255, -0.0159950, 0.0163251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169439, upper bound: 0.0171097
time: 1.87 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169439, upper bound: 0.0174123
time: 2.22 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0064495, 0.0047144, -0.0055383, 0.0035215, -0.0099710, 0.0102526
1: 0.9892128, 1.0098214, 0.9895932, 1.0080122, -0.0187994, 0.0202282
2: -0.0134823, 0.0055414, -0.0128205, 0.0045977, -0.0175411, 0.0178346
3: -0.0000912, 0.0058180, 0.0001639, 0.0056602, -0.0057513, 0.0056540
4: -0.0066347, 0.0090726, -0.0054354, 0.0085496, -0.0151843, 0.0145080
5: -0.0021577, 0.0107363, -0.0017427, 0.0103666, -0.0125242, 0.0124789
6: -0.0086142, 0.0032874, -0.0067616, 0.0027685, -0.0113826, 0.0100490
7: -0.0113428, -0.0000752, -0.0110766, -0.0007362, -0.0106067, 0.0110014
8: -0.0132840, 0.0159656, -0.0123602, 0.0150961, -0.0281985, 0.0281390
9: -0.0093816, 0.0074561, -0.0088987, 0.0069141, -0.0162957, 0.0163547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174844, upper bound: 0.0173931
time: 1.85 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174844, upper bound: 0.0177373
time: 2.01 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0056640, 0.0036861, -0.0049258, 0.0028024, -0.0084664, 0.0086120
1: 0.9890331, 1.0082619, 0.9893954, 1.0067960, -0.0177629, 0.0188665
2: -0.0137951, 0.0047280, -0.0131645, 0.0039634, -0.0172732, 0.0173829
3: 0.0001287, 0.0058926, 0.0003354, 0.0057422, -0.0056135, 0.0055572
4: -0.0056009, 0.0093198, -0.0046293, 0.0088215, -0.0144225, 0.0139491
5: -0.0018000, 0.0109110, -0.0014637, 0.0105588, -0.0123587, 0.0123748
6: -0.0070174, 0.0035326, -0.0055165, 0.0030383, -0.0100556, 0.0090492
7: -0.0114687, -0.0006449, -0.0112150, -0.0011804, -0.0102883, 0.0105701
8: -0.0124878, 0.0163766, -0.0117394, 0.0155482, -0.0278652, 0.0279480
9: -0.0096098, 0.0069889, -0.0091497, 0.0065499, -0.0161597, 0.0161387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174202, upper bound: 0.0172411
time: 2.40 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174202, upper bound: 0.0172410
time: 1.86 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0062249, 0.0044203, -0.0052138, 0.0030967, -0.0093215, 0.0096341
1: 0.9891161, 1.0093753, 0.9893862, 1.0073677, -0.0182517, 0.0199891
2: -0.0136506, 0.0053088, -0.0131807, 0.0042616, -0.0174350, 0.0179483
3: -0.0000283, 0.0058581, 0.0002548, 0.0057461, -0.0057743, 0.0056033
4: -0.0063390, 0.0092057, -0.0050082, 0.0088343, -0.0151734, 0.0142139
5: -0.0020554, 0.0108303, -0.0015949, 0.0105678, -0.0126232, 0.0124252
6: -0.0081575, 0.0034194, -0.0061019, 0.0030510, -0.0112085, 0.0095212
7: -0.0114106, -0.0002381, -0.0112216, -0.0009716, -0.0104390, 0.0109834
8: -0.0130563, 0.0161868, -0.0120313, 0.0155695, -0.0284455, 0.0280531
9: -0.0095044, 0.0073225, -0.0091616, 0.0067211, -0.0162255, 0.0164841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176675, upper bound: 0.0177280
time: 2.18 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176675, upper bound: 0.0177282
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0055462, 0.0035319, -0.0053900, 0.0033274, -0.0088736, 0.0089219
1: 0.9891343, 1.0080278, 0.9895229, 1.0077176, -0.0185833, 0.0185049
2: -0.0136190, 0.0046059, -0.0129429, 0.0044442, -0.0175746, 0.0170766
3: 0.0001617, 0.0058506, 0.0002054, 0.0056894, -0.0055277, 0.0056451
4: -0.0054458, 0.0091807, -0.0052402, 0.0086464, -0.0140922, 0.0144209
5: -0.0017463, 0.0108127, -0.0016752, 0.0104350, -0.0121813, 0.0124879
6: -0.0067777, 0.0033946, -0.0064603, 0.0028645, -0.0096423, 0.0098549
7: -0.0113979, -0.0007304, -0.0111259, -0.0008437, -0.0105542, 0.0103955
8: -0.0123683, 0.0161453, -0.0122100, 0.0152570, -0.0274600, 0.0281844
9: -0.0094813, 0.0069188, -0.0089881, 0.0068260, -0.0163073, 0.0159069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172477, upper bound: 0.0169096
time: 2.06 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172477, upper bound: 0.0169096
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0061056, 0.0042642, -0.0056759, 0.0037016, -0.0098072, 0.0099401
1: 0.9892222, 1.0091385, 0.9895133, 1.0082852, -0.0190629, 0.0196251
2: -0.0134658, 0.0051853, -0.0129595, 0.0047402, -0.0177281, 0.0176399
3: 0.0000051, 0.0058140, 0.0001254, 0.0056933, -0.0056882, 0.0056886
4: -0.0061821, 0.0090596, -0.0056164, 0.0086595, -0.0148416, 0.0146760
5: -0.0020011, 0.0107271, -0.0018053, 0.0104443, -0.0124453, 0.0125324
6: -0.0079151, 0.0032745, -0.0070413, 0.0028775, -0.0107926, 0.0103158
7: -0.0113362, -0.0003246, -0.0111325, -0.0006364, -0.0106999, 0.0108079
8: -0.0129354, 0.0159440, -0.0124997, 0.0152788, -0.0280401, 0.0282754
9: -0.0093695, 0.0072516, -0.0090001, 0.0069959, -0.0163655, 0.0162517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174331, upper bound: 0.0173812
time: 2.27 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174331, upper bound: 0.0173805
time: 2.22 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0058066, 0.0038728, -0.0051087, 0.0030031, -0.0088097, 0.0089815
1: 0.9888515, 1.0085448, 0.9888892, 1.0071592, -0.0183077, 0.0196556
2: -0.0141105, 0.0048756, -0.0140451, 0.0041528, -0.0177655, 0.0182792
3: 0.0000888, 0.0059678, 0.0002842, 0.0059522, -0.0058634, 0.0056836
4: -0.0057885, 0.0095692, -0.0048700, 0.0095175, -0.0153061, 0.0144391
5: -0.0018649, 0.0110873, -0.0015470, 0.0110508, -0.0129157, 0.0126343
6: -0.0073072, 0.0037800, -0.0058883, 0.0037288, -0.0110360, 0.0096683
7: -0.0115956, -0.0005415, -0.0115693, -0.0010477, -0.0105478, 0.0110278
8: -0.0126323, 0.0167911, -0.0119248, 0.0167052, -0.0291363, 0.0285426
9: -0.0098399, 0.0070737, -0.0097923, 0.0066587, -0.0164986, 0.0168660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173327, upper bound: 0.0175095
time: 1.65 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173327, upper bound: 0.0174500
time: 1.79 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0063734, 0.0046148, -0.0054089, 0.0033521, -0.0097256, 0.0100237
1: 0.9889280, 1.0096704, 0.9888796, 1.0077552, -0.0188272, 0.0207908
2: -0.0139779, 0.0054626, -0.0140618, 0.0044637, -0.0179477, 0.0188688
3: -0.0000699, 0.0059362, 0.0002002, 0.0059562, -0.0060260, 0.0057360
4: -0.0065346, 0.0094644, -0.0052651, 0.0095307, -0.0160653, 0.0147294
5: -0.0021230, 0.0110133, -0.0016837, 0.0110601, -0.0131831, 0.0126970
6: -0.0084596, 0.0036760, -0.0064986, 0.0037418, -0.0122014, 0.0101746
7: -0.0115423, -0.0001303, -0.0115760, -0.0008300, -0.0107123, 0.0114457
8: -0.0132070, 0.0166169, -0.0122291, 0.0167270, -0.0297235, 0.0286710
9: -0.0097432, 0.0074109, -0.0098044, 0.0068372, -0.0165804, 0.0172152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176159, upper bound: 0.0180297
time: 2.14 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176159, upper bound: 0.0178154
time: 1.98 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0058066, 0.0038728, -0.0048608, 0.0030498, -0.0088564, 0.0087336
1: 0.9888515, 1.0085448, 0.9887716, 1.0066668, -0.0178152, 0.0197732
2: -0.0141105, 0.0048756, -0.0142499, 0.0038960, -0.0175554, 0.0185116
3: 0.0000888, 0.0059678, 0.0003536, 0.0060010, -0.0059122, 0.0056142
4: -0.0057885, 0.0095692, -0.0045436, 0.0096793, -0.0154679, 0.0141128
5: -0.0018649, 0.0110873, -0.0014341, 0.0111652, -0.0130301, 0.0125214
6: -0.0073072, 0.0037800, -0.0053842, 0.0038893, -0.0111965, 0.0091642
7: -0.0115956, -0.0005415, -0.0116517, -0.0012276, -0.0103680, 0.0111102
8: -0.0126323, 0.0167911, -0.0116734, 0.0169742, -0.0294110, 0.0283061
9: -0.0098399, 0.0070737, -0.0099416, 0.0065112, -0.0163511, 0.0170153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172412, upper bound: 0.0174694
time: 1.62 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172412, upper bound: 0.0174118
time: 1.89 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0063734, 0.0046148, -0.0051655, 0.0030536, -0.0094271, 0.0097803
1: 0.9889280, 1.0096704, 0.9887621, 1.0072719, -0.0183439, 0.0209083
2: -0.0139779, 0.0054626, -0.0142664, 0.0042116, -0.0177385, 0.0191067
3: -0.0000699, 0.0059362, 0.0002683, 0.0060050, -0.0060748, 0.0056678
4: -0.0065346, 0.0094644, -0.0049447, 0.0096924, -0.0162270, 0.0144091
5: -0.0021230, 0.0110133, -0.0015729, 0.0111744, -0.0132975, 0.0125862
6: -0.0084596, 0.0036760, -0.0060037, 0.0039023, -0.0123619, 0.0096797
7: -0.0115423, -0.0001303, -0.0116583, -0.0010066, -0.0105357, 0.0115280
8: -0.0132070, 0.0166169, -0.0119823, 0.0169959, -0.0299977, 0.0284386
9: -0.0097432, 0.0074109, -0.0099537, 0.0066924, -0.0164356, 0.0173646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175428, upper bound: 0.0179932
time: 2.40 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175428, upper bound: 0.0177793
time: 1.71 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0064850, 0.0047609, -0.0058581, 0.0039401, -0.0104252, 0.0106190
1: 0.9897188, 1.0098919, 0.9890032, 1.0086468, -0.0189281, 0.0208887
2: -0.0126021, 0.0055782, -0.0138470, 0.0049289, -0.0170536, 0.0188749
3: -0.0001011, 0.0056081, 0.0000744, 0.0059049, -0.0060061, 0.0055337
4: -0.0066815, 0.0083770, -0.0058562, 0.0093609, -0.0160424, 0.0142333
5: -0.0021739, 0.0102445, -0.0018883, 0.0109401, -0.0131140, 0.0121328
6: -0.0086865, 0.0025973, -0.0074118, 0.0035734, -0.0122599, 0.0100090
7: -0.0109888, -0.0000494, -0.0114896, -0.0005042, -0.0104846, 0.0114402
8: -0.0133201, 0.0148092, -0.0126845, 0.0164448, -0.0295903, 0.0273162
9: -0.0087394, 0.0074772, -0.0096476, 0.0071043, -0.0158437, 0.0171249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170998, upper bound: 0.0174203
time: 2.14 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173476, upper bound: 0.0179684
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0064850, 0.0047609, -0.0056034, 0.0036068, -0.0100918, 0.0103644
1: 0.9897188, 1.0098919, 0.9889076, 1.0081414, -0.0184226, 0.0209842
2: -0.0126021, 0.0055782, -0.0140132, 0.0046652, -0.0168339, 0.0190837
3: -0.0001011, 0.0056081, 0.0001457, 0.0059446, -0.0060457, 0.0054624
4: -0.0066815, 0.0083770, -0.0055211, 0.0094923, -0.0161738, 0.0138982
5: -0.0021739, 0.0102445, -0.0017723, 0.0110330, -0.0132069, 0.0120169
6: -0.0086865, 0.0025973, -0.0068941, 0.0037037, -0.0123903, 0.0094914
7: -0.0109888, -0.0000494, -0.0115564, -0.0006889, -0.0102999, 0.0115071
8: -0.0133201, 0.0148092, -0.0124263, 0.0166633, -0.0298169, 0.0270670
9: -0.0087394, 0.0074772, -0.0097690, 0.0069529, -0.0156923, 0.0172462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170998, upper bound: 0.0174203
time: 1.77 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173476, upper bound: 0.0179684
time: 2.29 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0064888, 0.0047659, -0.0058581, 0.0039401, -0.0104289, 0.0106239
1: 0.9891075, 1.0098995, 0.9890032, 1.0086468, -0.0195393, 0.0208963
2: -0.0136654, 0.0055822, -0.0138470, 0.0049289, -0.0180111, 0.0188039
3: -0.0001022, 0.0058617, 0.0000744, 0.0059049, -0.0060071, 0.0057872
4: -0.0066865, 0.0092174, -0.0058562, 0.0093609, -0.0160474, 0.0150737
5: -0.0021756, 0.0108387, -0.0018883, 0.0109401, -0.0131157, 0.0127270
6: -0.0086942, 0.0034310, -0.0074118, 0.0035734, -0.0122676, 0.0108428
7: -0.0114166, -0.0000466, -0.0114896, -0.0005042, -0.0109124, 0.0114429
8: -0.0133240, 0.0162063, -0.0126845, 0.0164448, -0.0295689, 0.0286931
9: -0.0095152, 0.0074795, -0.0096476, 0.0071043, -0.0166195, 0.0171271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173284, upper bound: 0.0173854
time: 2.00 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174581, upper bound: 0.0177686
time: 2.24 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0064888, 0.0047659, -0.0056034, 0.0036068, -0.0100956, 0.0103693
1: 0.9891075, 1.0098995, 0.9889076, 1.0081414, -0.0190338, 0.0209919
2: -0.0136654, 0.0055822, -0.0140132, 0.0046652, -0.0177923, 0.0190110
3: -0.0001022, 0.0058617, 0.0001457, 0.0059446, -0.0060468, 0.0057160
4: -0.0066865, 0.0092174, -0.0055211, 0.0094923, -0.0161788, 0.0147386
5: -0.0021756, 0.0108387, -0.0017723, 0.0110330, -0.0132086, 0.0126110
6: -0.0086942, 0.0034310, -0.0068941, 0.0037037, -0.0123980, 0.0103252
7: -0.0114166, -0.0000466, -0.0115564, -0.0006889, -0.0107277, 0.0115098
8: -0.0133240, 0.0162063, -0.0124263, 0.0166633, -0.0297943, 0.0284444
9: -0.0095152, 0.0074795, -0.0097690, 0.0069529, -0.0164681, 0.0172485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173284, upper bound: 0.0173854
time: 1.93 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174581, upper bound: 0.0177686
time: 2.34 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0054238, 0.0033716, -0.0066859, 0.0050238, -0.0104476, 0.0100575
1: 0.9894825, 1.0077846, 0.9891016, 1.0102907, -0.0208082, 0.0186830
2: -0.0130131, 0.0044791, -0.0136760, 0.0057862, -0.0182837, 0.0176453
3: 0.0001960, 0.0057061, -0.0001573, 0.0058642, -0.0056682, 0.0058634
4: -0.0052847, 0.0087018, -0.0069458, 0.0092257, -0.0145104, 0.0156476
5: -0.0016905, 0.0104742, -0.0022653, 0.0108445, -0.0125351, 0.0127395
6: -0.0065289, 0.0029195, -0.0090948, 0.0034393, -0.0099682, 0.0120143
7: -0.0111541, -0.0008192, -0.0114208, 0.0000963, -0.0112503, 0.0106016
8: -0.0122442, 0.0153491, -0.0135237, 0.0162201, -0.0282874, 0.0286892
9: -0.0090392, 0.0068460, -0.0095229, 0.0075967, -0.0166359, 0.0163689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172726, upper bound: 0.0175036
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177438, upper bound: 0.0177565
time: 2.21 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0059039, 0.0040002, -0.0065571, 0.0048552, -0.0107592, 0.0105572
1: 0.9895843, 1.0087380, 0.9892079, 1.0100350, -0.0204507, 0.0195302
2: -0.0128359, 0.0049764, -0.0134910, 0.0056528, -0.0179911, 0.0179507
3: 0.0000616, 0.0056639, -0.0001213, 0.0058200, -0.0057585, 0.0057851
4: -0.0059166, 0.0085618, -0.0067763, 0.0090795, -0.0149962, 0.0153381
5: -0.0019092, 0.0103751, -0.0022067, 0.0107412, -0.0126504, 0.0125818
6: -0.0075050, 0.0027805, -0.0088330, 0.0032942, -0.0107993, 0.0116135
7: -0.0110828, -0.0004709, -0.0113463, 0.0000029, -0.0110856, 0.0108754
8: -0.0127310, 0.0151163, -0.0133931, 0.0159771, -0.0285291, 0.0283324
9: -0.0089099, 0.0071316, -0.0093879, 0.0075201, -0.0164300, 0.0165195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172476, upper bound: 0.0173202
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177373, upper bound: 0.0174902
time: 2.26 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0054238, 0.0033716, -0.0065038, 0.0047854, -0.0102092, 0.0098754
1: 0.9894825, 1.0077846, 0.9889925, 1.0099292, -0.0204467, 0.0187922
2: -0.0130131, 0.0044791, -0.0138658, 0.0055976, -0.0181348, 0.0178706
3: 0.0001960, 0.0057061, -0.0001064, 0.0059094, -0.0057134, 0.0058125
4: -0.0052847, 0.0087018, -0.0067061, 0.0093757, -0.0146604, 0.0154080
5: -0.0016905, 0.0104742, -0.0021824, 0.0109506, -0.0126411, 0.0126566
6: -0.0065289, 0.0029195, -0.0087246, 0.0035881, -0.0101170, 0.0116441
7: -0.0111541, -0.0008192, -0.0114971, -0.0000358, -0.0111183, 0.0106780
8: -0.0122442, 0.0153491, -0.0133391, 0.0164695, -0.0285437, 0.0285160
9: -0.0090392, 0.0068460, -0.0096614, 0.0074884, -0.0165276, 0.0165074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172162, upper bound: 0.0174839
time: 1.90 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176960, upper bound: 0.0177333
time: 1.96 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0059039, 0.0040002, -0.0063789, 0.0046219, -0.0105259, 0.0103790
1: 0.9895843, 1.0087380, 0.9891014, 1.0096813, -0.0200970, 0.0196367
2: -0.0128359, 0.0049764, -0.0136762, 0.0054683, -0.0178458, 0.0181705
3: 0.0000616, 0.0056639, -0.0000714, 0.0058642, -0.0058026, 0.0057352
4: -0.0059166, 0.0085618, -0.0065418, 0.0092259, -0.0151426, 0.0151035
5: -0.0019092, 0.0103751, -0.0021255, 0.0108447, -0.0127539, 0.0125007
6: -0.0075050, 0.0027805, -0.0084706, 0.0034395, -0.0109445, 0.0112512
7: -0.0110828, -0.0004709, -0.0114209, -0.0001264, -0.0109564, 0.0109500
8: -0.0127310, 0.0151163, -0.0132125, 0.0162205, -0.0287812, 0.0281627
9: -0.0089099, 0.0071316, -0.0095231, 0.0074141, -0.0163240, 0.0166547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173713, upper bound: 0.0169360
time: 2.20 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176856, upper bound: 0.0174823
time: 1.80 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0053935, 0.0033320, -0.0068428, 0.0052293, -0.0106228, 0.0101748
1: 0.9888794, 1.0077246, 0.9889134, 1.0106025, -0.0217231, 0.0188112
2: -0.0140621, 0.0044478, -0.0140034, 0.0059488, -0.0194064, 0.0179671
3: 0.0002045, 0.0059562, -0.0002013, 0.0059422, -0.0057378, 0.0061575
4: -0.0052448, 0.0095309, -0.0071524, 0.0094845, -0.0147293, 0.0166834
5: -0.0016767, 0.0110603, -0.0023368, 0.0110275, -0.0127042, 0.0133971
6: -0.0064674, 0.0037421, -0.0094139, 0.0036960, -0.0101634, 0.0131560
7: -0.0115761, -0.0008412, -0.0115525, 0.0002102, -0.0117863, 0.0107114
8: -0.0122135, 0.0167275, -0.0136829, 0.0166503, -0.0286885, 0.0302147
9: -0.0098047, 0.0068280, -0.0097618, 0.0076900, -0.0174947, 0.0165898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175771, upper bound: 0.0176268
time: 1.90 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180365, upper bound: 0.0178423
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058371, 0.0039128, -0.0067138, 0.0050604, -0.0108975, 0.0106265
1: 0.9890032, 1.0086055, 0.9890206, 1.0103459, -0.0213428, 0.0195848
2: -0.0138470, 0.0049072, -0.0138166, 0.0058151, -0.0190785, 0.0182318
3: 0.0000803, 0.0059049, -0.0001651, 0.0058977, -0.0058174, 0.0060701
4: -0.0058287, 0.0093609, -0.0069825, 0.0093369, -0.0151656, 0.0163434
5: -0.0018788, 0.0109401, -0.0022781, 0.0109232, -0.0128020, 0.0132181
6: -0.0073693, 0.0035734, -0.0091515, 0.0035496, -0.0109188, 0.0127249
7: -0.0114896, -0.0005194, -0.0114774, 0.0001165, -0.0116061, 0.0109580
8: -0.0126632, 0.0164448, -0.0135520, 0.0164050, -0.0288938, 0.0298034
9: -0.0096476, 0.0070919, -0.0096255, 0.0076133, -0.0172609, 0.0167174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175442, upper bound: 0.0174240
time: 2.07 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180090, upper bound: 0.0175419
time: 1.64 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0053935, 0.0033320, -0.0066576, 0.0049868, -0.0103803, 0.0099896
1: 0.9888794, 1.0077246, 0.9888048, 1.0102346, -0.0213552, 0.0189198
2: -0.0140621, 0.0044478, -0.0141923, 0.0057569, -0.0192551, 0.0181936
3: 0.0002045, 0.0059562, -0.0001494, 0.0059873, -0.0057828, 0.0061056
4: -0.0052448, 0.0095309, -0.0069086, 0.0096338, -0.0148787, 0.0164395
5: -0.0016767, 0.0110603, -0.0022525, 0.0111330, -0.0128098, 0.0133128
6: -0.0064674, 0.0037421, -0.0090373, 0.0038442, -0.0103115, 0.0127794
7: -0.0115761, -0.0008412, -0.0116285, 0.0000758, -0.0116519, 0.0107873
8: -0.0122135, 0.0167275, -0.0134950, 0.0168986, -0.0289477, 0.0300387
9: -0.0098047, 0.0068280, -0.0098996, 0.0075799, -0.0173845, 0.0167277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175041, upper bound: 0.0175987
time: 2.23 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179653, upper bound: 0.0177959
time: 1.98 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058371, 0.0039128, -0.0065314, 0.0048216, -0.0106588, 0.0104442
1: 0.9890032, 1.0086055, 0.9889138, 1.0099840, -0.0209808, 0.0196917
2: -0.0138470, 0.0049072, -0.0140024, 0.0056262, -0.0189305, 0.0184537
3: 0.0000803, 0.0059049, -0.0001141, 0.0059420, -0.0058617, 0.0060190
4: -0.0058287, 0.0093609, -0.0067425, 0.0094838, -0.0153125, 0.0161034
5: -0.0018788, 0.0109401, -0.0021950, 0.0110270, -0.0129057, 0.0131351
6: -0.0073693, 0.0035734, -0.0087808, 0.0036953, -0.0110645, 0.0123541
7: -0.0114896, -0.0005194, -0.0115521, -0.0000158, -0.0114738, 0.0110327
8: -0.0126632, 0.0164448, -0.0133671, 0.0166491, -0.0291485, 0.0296299
9: -0.0096476, 0.0070919, -0.0097611, 0.0075048, -0.0171524, 0.0168530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174632, upper bound: 0.0174004
time: 3.96 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179369, upper bound: 0.0175235
time: 2.74 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0067018, 0.0050447, -0.0064939, 0.0047726, -0.0114744, 0.0115386
1: 0.9896147, 1.0103225, 0.9892662, 1.0099096, -0.0202949, 0.0210563
2: -0.0127832, 0.0058027, -0.0133896, 0.0055874, -0.0178576, 0.0186341
3: -0.0001618, 0.0056513, -0.0001036, 0.0057959, -0.0059577, 0.0057549
4: -0.0069668, 0.0085202, -0.0066932, 0.0089994, -0.0159662, 0.0152134
5: -0.0022726, 0.0103457, -0.0021779, 0.0106846, -0.0129572, 0.0125237
6: -0.0091272, 0.0027393, -0.0087046, 0.0032148, -0.0123419, 0.0114439
7: -0.0110616, 0.0001078, -0.0113056, -0.0000429, -0.0110187, 0.0114134
8: -0.0135399, 0.0150472, -0.0133291, 0.0158439, -0.0292001, 0.0281933
9: -0.0088715, 0.0076061, -0.0093140, 0.0074825, -0.0163540, 0.0169201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179477, upper bound: 0.0178863
time: 3.23 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0181893, upper bound: 0.0184744
time: 2.03 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0065832, 0.0048895, -0.0068757, 0.0052723, -0.0118555, 0.0117652
1: 0.9897140, 1.0100870, 0.9894072, 1.0106676, -0.0209536, 0.0206798
2: -0.0126104, 0.0056799, -0.0131441, 0.0059828, -0.0180718, 0.0183066
3: -0.0001286, 0.0056101, -0.0002105, 0.0057373, -0.0058659, 0.0058205
4: -0.0068107, 0.0083836, -0.0071957, 0.0088053, -0.0156161, 0.0155792
5: -0.0022186, 0.0102492, -0.0023518, 0.0105474, -0.0127659, 0.0126010
6: -0.0088861, 0.0026038, -0.0094807, 0.0030222, -0.0119083, 0.0120845
7: -0.0109921, 0.0000218, -0.0112068, 0.0002340, -0.0112260, 0.0112286
8: -0.0134197, 0.0148201, -0.0137162, 0.0155213, -0.0287615, 0.0283515
9: -0.0087454, 0.0075356, -0.0091348, 0.0077096, -0.0164550, 0.0166704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177247, upper bound: 0.0182327
time: 2.22 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173931, upper bound: 0.0184612
time: 2.04 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0067018, 0.0050447, -0.0063059, 0.0045264, -0.0112282, 0.0113506
1: 0.9896147, 1.0103225, 0.9891515, 1.0095363, -0.0199215, 0.0211709
2: -0.0127832, 0.0058027, -0.0135888, 0.0053927, -0.0177246, 0.0188673
3: -0.0001618, 0.0056513, -0.0000509, 0.0058434, -0.0060052, 0.0057022
4: -0.0069668, 0.0085202, -0.0064457, 0.0091568, -0.0161236, 0.0149658
5: -0.0022726, 0.0103457, -0.0020923, 0.0107959, -0.0130684, 0.0124380
6: -0.0091272, 0.0027393, -0.0083223, 0.0033709, -0.0124981, 0.0110615
7: -0.0110616, 0.0001078, -0.0113857, -0.0001793, -0.0108823, 0.0114936
8: -0.0135399, 0.0150472, -0.0131385, 0.0161056, -0.0294706, 0.0280133
9: -0.0088715, 0.0076061, -0.0094593, 0.0073707, -0.0162422, 0.0170654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178748, upper bound: 0.0178801
time: 2.27 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0181148, upper bound: 0.0184556
time: 2.49 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0065832, 0.0048895, -0.0066689, 0.0050016, -0.0115848, 0.0115583
1: 0.9897140, 1.0100870, 0.9893183, 1.0102569, -0.0205429, 0.0207688
2: -0.0126104, 0.0056799, -0.0132989, 0.0057686, -0.0179236, 0.0184995
3: -0.0001286, 0.0056101, -0.0001526, 0.0057743, -0.0059029, 0.0057627
4: -0.0068107, 0.0083836, -0.0069235, 0.0089277, -0.0157385, 0.0153070
5: -0.0022186, 0.0102492, -0.0022576, 0.0106339, -0.0128525, 0.0125068
6: -0.0088861, 0.0026038, -0.0090603, 0.0031437, -0.0120297, 0.0116640
7: -0.0109921, 0.0000218, -0.0112691, 0.0000840, -0.0110760, 0.0112909
8: -0.0134197, 0.0148201, -0.0135065, 0.0157247, -0.0289789, 0.0281496
9: -0.0087454, 0.0075356, -0.0092478, 0.0075866, -0.0163320, 0.0167834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176863, upper bound: 0.0182189
time: 2.39 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0180748, upper bound: 0.0184456
time: 2.60 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0067112, 0.0050570, -0.0067018, 0.0050447, -0.0117559, 0.0117587
1: 0.9889947, 1.0103409, 0.9896147, 1.0103225, -0.0213278, 0.0207262
2: -0.0138617, 0.0058124, -0.0127832, 0.0058027, -0.0191034, 0.0181066
3: -0.0001644, 0.0059084, -0.0001618, 0.0056513, -0.0058157, 0.0060702
4: -0.0069791, 0.0093725, -0.0069668, 0.0085202, -0.0154993, 0.0163393
5: -0.0022769, 0.0109483, -0.0022726, 0.0103457, -0.0126226, 0.0132209
6: -0.0091462, 0.0035849, -0.0091272, 0.0027393, -0.0118855, 0.0127121
7: -0.0114955, 0.0001146, -0.0110616, 0.0001078, -0.0116033, 0.0111763
8: -0.0135494, 0.0164641, -0.0135399, 0.0150472, -0.0284137, 0.0298206
9: -0.0096584, 0.0076117, -0.0088715, 0.0076061, -0.0172645, 0.0164832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0185436, upper bound: 0.0183233
time: 2.98 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0185380, upper bound: 0.0182778
time: 2.69 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0067112, 0.0050570, -0.0067112, 0.0050570, -0.0117681, 0.0117681
1: 0.9889947, 1.0103409, 0.9889947, 1.0103409, -0.0213463, 0.0213463
2: -0.0138617, 0.0058124, -0.0138617, 0.0058124, -0.0190777, 0.0190777
3: -0.0001644, 0.0059084, -0.0001644, 0.0059084, -0.0060729, 0.0060729
4: -0.0069791, 0.0093725, -0.0069791, 0.0093725, -0.0163516, 0.0163516
5: -0.0022769, 0.0109483, -0.0022769, 0.0109483, -0.0132252, 0.0132252
6: -0.0091462, 0.0035849, -0.0091462, 0.0035849, -0.0127312, 0.0127312
7: -0.0114955, 0.0001146, -0.0114955, 0.0001146, -0.0116101, 0.0116101
8: -0.0135494, 0.0164641, -0.0135494, 0.0164641, -0.0298100, 0.0298100
9: -0.0096584, 0.0076117, -0.0096584, 0.0076117, -0.0172701, 0.0172701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0185436, upper bound: 0.0183826
time: 1.93 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0185380, upper bound: 0.0183272
time: 2.19 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0067112, 0.0050570, -0.0065028, 0.0047842, -0.0114953, 0.0115597
1: 0.9889947, 1.0103409, 0.9895087, 1.0099273, -0.0209326, 0.0208322
2: -0.0138617, 0.0058124, -0.0129676, 0.0055966, -0.0189641, 0.0183268
3: -0.0001644, 0.0059084, -0.0001061, 0.0056953, -0.0058597, 0.0060145
4: -0.0069791, 0.0093725, -0.0067048, 0.0086659, -0.0156450, 0.0160773
5: -0.0022769, 0.0109483, -0.0021819, 0.0104488, -0.0127256, 0.0131303
6: -0.0091462, 0.0035849, -0.0087226, 0.0028839, -0.0120301, 0.0123075
7: -0.0114955, 0.0001146, -0.0111358, -0.0000365, -0.0114589, 0.0112504
8: -0.0135494, 0.0164641, -0.0133381, 0.0152895, -0.0286682, 0.0296291
9: -0.0096584, 0.0076117, -0.0090061, 0.0074878, -0.0171462, 0.0166178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0185082, upper bound: 0.0183153
time: 2.07 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184994, upper bound: 0.0182669
time: 2.05 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0067112, 0.0050570, -0.0065247, 0.0048129, -0.0115240, 0.0115817
1: 0.9889947, 1.0103409, 0.9888845, 1.0099707, -0.0209760, 0.0214564
2: -0.0138617, 0.0058124, -0.0140534, 0.0056193, -0.0189214, 0.0193081
3: -0.0001644, 0.0059084, -0.0001122, 0.0059542, -0.0061186, 0.0060207
4: -0.0069791, 0.0093725, -0.0067337, 0.0095240, -0.0165032, 0.0161062
5: -0.0022769, 0.0109483, -0.0021919, 0.0110554, -0.0133323, 0.0131402
6: -0.0091462, 0.0035849, -0.0087671, 0.0037353, -0.0128815, 0.0123520
7: -0.0114955, 0.0001146, -0.0115726, -0.0000206, -0.0114749, 0.0116873
8: -0.0135494, 0.0164641, -0.0133603, 0.0167161, -0.0300757, 0.0296311
9: -0.0096584, 0.0076117, -0.0097983, 0.0075008, -0.0171592, 0.0174100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0185443, upper bound: 0.0182993
time: 2.46 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0184994, upper bound: 0.0183116
time: 2.29 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0052515, 0.0031461, -0.0053297, 0.0032484, -0.0084999, 0.0084757
1: 0.9890762, 1.0074426, 0.9893128, 1.0075976, -0.0185214, 0.0181298
2: -0.0137200, 0.0043007, -0.0133086, 0.0043817, -0.0176039, 0.0170967
3: 0.0002442, 0.0058747, 0.0002223, 0.0057766, -0.0055324, 0.0056523
4: -0.0050579, 0.0092605, -0.0051608, 0.0089354, -0.0139933, 0.0144213
5: -0.0016121, 0.0108691, -0.0016477, 0.0106393, -0.0122513, 0.0125168
6: -0.0061786, 0.0034738, -0.0063375, 0.0031512, -0.0093298, 0.0098113
7: -0.0114385, -0.0009442, -0.0112730, -0.0008875, -0.0105510, 0.0103288
8: -0.0120695, 0.0162779, -0.0121488, 0.0157375, -0.0276376, 0.0282582
9: -0.0095550, 0.0067436, -0.0092549, 0.0067901, -0.0163450, 0.0159984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173152, upper bound: 0.0173251
time: 1.89 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173152, upper bound: 0.0173252
time: 1.85 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0052515, 0.0031461, -0.0051049, 0.0029541, -0.0082056, 0.0082509
1: 0.9890762, 1.0074426, 0.9892251, 1.0071516, -0.0180754, 0.0182174
2: -0.0137200, 0.0043007, -0.0134608, 0.0041488, -0.0173506, 0.0172344
3: 0.0002442, 0.0058747, 0.0002853, 0.0058129, -0.0055687, 0.0055894
4: -0.0050579, 0.0092605, -0.0048649, 0.0090557, -0.0141136, 0.0141254
5: -0.0016121, 0.0108691, -0.0015453, 0.0107243, -0.0123364, 0.0124144
6: -0.0061786, 0.0034738, -0.0058805, 0.0032706, -0.0094492, 0.0093543
7: -0.0114385, -0.0009442, -0.0113342, -0.0010505, -0.0103880, 0.0103901
8: -0.0120695, 0.0162779, -0.0119209, 0.0159375, -0.0278387, 0.0280302
9: -0.0095550, 0.0067436, -0.0093659, 0.0066564, -0.0162113, 0.0161095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173152, upper bound: 0.0173252
time: 3.35 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173152, upper bound: 0.0173252
time: 2.00 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058134, 0.0038817, -0.0056099, 0.0036153, -0.0094287, 0.0094916
1: 0.9891667, 1.0085585, 0.9893035, 1.0081544, -0.0189877, 0.0192550
2: -0.0135627, 0.0048827, -0.0133246, 0.0046719, -0.0177452, 0.0176595
3: 0.0000869, 0.0058372, 0.0001439, 0.0057804, -0.0056935, 0.0056933
4: -0.0057975, 0.0091362, -0.0055296, 0.0089480, -0.0147455, 0.0146658
5: -0.0018680, 0.0107813, -0.0017753, 0.0106482, -0.0125162, 0.0125565
6: -0.0073211, 0.0033505, -0.0069072, 0.0031638, -0.0104848, 0.0102577
7: -0.0113752, -0.0005365, -0.0112794, -0.0006842, -0.0106910, 0.0107429
8: -0.0126392, 0.0160713, -0.0124329, 0.0157585, -0.0282140, 0.0283341
9: -0.0094402, 0.0070778, -0.0092665, 0.0069567, -0.0163970, 0.0163443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177446
time: 4.31 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177447
time: 2.47 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058134, 0.0038817, -0.0053926, 0.0033308, -0.0091442, 0.0092743
1: 0.9891667, 1.0085585, 0.9892159, 1.0077227, -0.0185561, 0.0193426
2: -0.0135627, 0.0048827, -0.0134769, 0.0044468, -0.0175011, 0.0177983
3: 0.0000869, 0.0058372, 0.0002047, 0.0058167, -0.0057298, 0.0056324
4: -0.0057975, 0.0091362, -0.0052436, 0.0090684, -0.0148660, 0.0143798
5: -0.0018680, 0.0107813, -0.0016763, 0.0107333, -0.0126013, 0.0124576
6: -0.0073211, 0.0033505, -0.0064654, 0.0032832, -0.0106043, 0.0098159
7: -0.0113752, -0.0005365, -0.0113407, -0.0008418, -0.0105334, 0.0108042
8: -0.0126392, 0.0160713, -0.0122126, 0.0159587, -0.0284172, 0.0281148
9: -0.0094402, 0.0070778, -0.0093777, 0.0068275, -0.0162677, 0.0164555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177447
time: 1.95 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174506, upper bound: 0.0177447
time: 1.88 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0056803, 0.0037075, -0.0052127, 0.0030953, -0.0087756, 0.0089202
1: 0.9892157, 1.0082941, 0.9894097, 1.0073656, -0.0181499, 0.0188844
2: -0.0134774, 0.0047448, -0.0131400, 0.0042605, -0.0172657, 0.0173576
3: 0.0001242, 0.0058168, 0.0002551, 0.0057364, -0.0056122, 0.0055617
4: -0.0056223, 0.0090688, -0.0050068, 0.0088022, -0.0144245, 0.0140756
5: -0.0018074, 0.0107336, -0.0015944, 0.0105451, -0.0123525, 0.0123279
6: -0.0070505, 0.0032836, -0.0060997, 0.0030191, -0.0100695, 0.0093833
7: -0.0113409, -0.0006331, -0.0112052, -0.0009723, -0.0103686, 0.0105721
8: -0.0125043, 0.0159592, -0.0120302, 0.0155160, -0.0278485, 0.0278237
9: -0.0093780, 0.0069986, -0.0091319, 0.0067205, -0.0160985, 0.0161305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.62 + 595.43 = 600.05 seconds
