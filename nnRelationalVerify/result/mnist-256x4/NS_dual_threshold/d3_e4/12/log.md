## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.6947652919999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548)
1: (-0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906)
2: (-0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625)
3: (-1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869)
4: (-1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024)
5: (-0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151)
6: (-0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975)
7: (-0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587)
8: (-1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157)
9: (-1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.04 + 2.86 = 3.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.25 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.49 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 2.49
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2, status: Status.UNKNOWN, split count: 1, time: 2.49
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -1.0862367, 0.9590441, -0.6399282, 0.7116064, -1.7978431, 1.5989723
1: -0.7331971, 0.7687183, -0.4611878, 0.4757072, -1.2089043, 1.2299061
2: -0.8073971, 1.0391001, -0.4399549, 0.7232603, -1.5306574, 1.4790549
3: -0.9127105, 0.7314767, -0.4073527, 0.5692190, -1.4819295, 1.1388295
4: -0.9891061, 0.9348162, -0.6087359, 0.5919434, -1.5810494, 1.5435522
5: -0.5973887, 1.2404648, -0.0838350, 1.1386771, -1.7360659, 1.3242998
6: -0.7891675, 0.9219666, -0.4896987, 0.6273942, -1.4165616, 1.4116653
7: -0.8390387, 0.9628478, -0.5218263, 0.6106682, -1.4497070, 1.4846741
8: -1.0786552, 0.9730691, -0.6580070, 0.7400061, -1.8186615, 1.6310761
9: -0.8919388, 0.9293619, -0.5616168, 0.6315347, -1.5234735, 1.4909787

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.22 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.18 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -1.2023056, 1.0259824, -1.1125209, 0.9741853, -2.1764908, 2.1385033
1: -0.8118370, 0.8530850, -0.7509660, 0.7879077, -1.5997446, 1.6040510
2: -0.9145819, 1.1174943, -0.8315364, 1.0569161, -1.9714980, 1.9490306
3: -1.0566809, 0.7754949, -0.9454460, 0.7413508, -1.7980318, 1.7209409
4: -1.1011605, 1.0273333, -1.0143080, 0.9556286, -2.0567892, 2.0416412
5: -0.7384169, 1.2762588, -0.6291003, 1.2485980, -1.9870149, 1.9053591
6: -0.8728874, 1.0135183, -0.8079731, 0.9426181, -1.8155055, 1.8214915
7: -0.9289262, 1.0567906, -0.8593305, 0.9841705, -1.9130967, 1.9161211
8: -1.1974562, 1.0373104, -1.1054493, 0.9873534, -2.1848097, 2.1427598
9: -0.9866645, 1.0090874, -0.9135427, 0.9472330, -1.9338975, 1.9226301

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.12 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.04 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -0.7762654, 0.7949417, -0.5938244, 0.6639912, -1.4402566, 1.3887661
1: -0.5408490, 0.5627488, -0.4226675, 0.4378105, -0.9786595, 0.9854163
2: -0.5431093, 0.8315909, -0.4061226, 0.6585521, -1.2016613, 1.2377135
3: -0.5496390, 0.6208208, -0.3704744, 0.5401527, -1.0897917, 0.9912952
4: -0.7140205, 0.6993443, -0.5649908, 0.5488838, -1.2629043, 1.2643350
5: -0.2365935, 1.1624922, -0.0102418, 1.1303495, -1.3669430, 1.1727340
6: -0.5769718, 0.7094610, -0.4571531, 0.5898031, -1.1667749, 1.1666141
7: -0.6204945, 0.7226583, -0.4848466, 0.5571934, -1.1776879, 1.2075049
8: -0.7763964, 0.8165095, -0.6122321, 0.6959898, -1.4723862, 1.4287416
9: -0.6583789, 0.7253534, -0.5188782, 0.5856174, -1.2439964, 1.2442316

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.24 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.24 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -1.5389435, 1.2202461, -0.6066934, 0.6788093, -2.2177529, 1.8269395
1: -1.0392663, 1.0997537, -0.4342327, 0.4490640, -1.4883304, 1.5339864
2: -1.2233183, 1.3464490, -0.4151252, 0.6785562, -1.9018745, 1.7615743
3: -1.4769297, 0.9018110, -0.3794613, 0.5492116, -2.0261412, 1.2812724
4: -1.4219991, 1.2938424, -0.5780947, 0.5614421, -1.9834412, 1.8719372
5: -1.1432135, 1.3807698, -0.0330769, 1.1325519, -2.2757654, 1.4138467
6: -1.1110196, 1.2782254, -0.4669446, 0.6011295, -1.7121491, 1.7451700
7: -1.1890962, 1.3306618, -0.4949140, 0.5737597, -1.7628559, 1.8255757
8: -1.5400405, 1.2194816, -0.6257544, 0.7096656, -2.2497060, 1.8452359
9: -1.2646285, 1.2373993, -0.5319730, 0.5992738, -1.8639023, 1.7693723

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.27 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.30 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -0.6399282, 0.7116064, -1.1125209, 0.9741853, -1.6141136, 1.8241273
1: -0.4611878, 0.4757072, -0.7509660, 0.7879077, -1.2490954, 1.2266732
2: -0.4399549, 0.7232603, -0.8315364, 1.0569161, -1.4968710, 1.5547967
3: -0.4073527, 0.5692190, -0.9454460, 0.7413508, -1.1487036, 1.5146650
4: -0.6087359, 0.5919434, -1.0143080, 0.9556286, -1.5643644, 1.6062515
5: -0.0838350, 1.1386771, -0.6291003, 1.2485980, -1.3324330, 1.7677774
6: -0.4896987, 0.6273942, -0.8079731, 0.9426181, -1.4323168, 1.4353673
7: -0.5218263, 0.6106682, -0.8593305, 0.9841705, -1.5059967, 1.4699988
8: -0.6580070, 0.7400061, -1.1054493, 0.9873534, -1.6453605, 1.8454554
9: -0.5616168, 0.6315347, -0.9135427, 0.9472330, -1.5088497, 1.5450773

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.15 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.33 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -1.1125209, 0.9741853, -1.1125209, 0.9741853, -2.0867062, 2.0867062
1: -0.7509660, 0.7879077, -0.7509660, 0.7879077, -1.5388737, 1.5388737
2: -0.8315364, 1.0569161, -0.8315364, 1.0569161, -1.8884525, 1.8884525
3: -0.9454460, 0.7413508, -0.9454460, 0.7413508, -1.6867969, 1.6867969
4: -1.0143080, 0.9556286, -1.0143080, 0.9556286, -1.9699366, 1.9699366
5: -0.6291003, 1.2485980, -0.6291003, 1.2485980, -1.8776983, 1.8776983
6: -0.8079731, 0.9426181, -0.8079731, 0.9426181, -1.7505913, 1.7505913
7: -0.8593305, 0.9841705, -0.8593305, 0.9841705, -1.8435011, 1.8435011
8: -1.1054493, 0.9873534, -1.1054493, 0.9873534, -2.0928028, 2.0928028
9: -0.9135427, 0.9472330, -0.9135427, 0.9472330, -1.8607757, 1.8607757

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 0.95 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.24 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.6207734, 0.6940692, -0.3694080, 0.4064409, -1.0272144, 1.0634772
1: -0.4466389, 0.4606532, -0.2287708, 0.2392205, -0.6858593, 0.6894240
2: -0.4251702, 0.6991562, -0.2285435, 0.3562610, -0.7814312, 0.9276996
3: -0.3896066, 0.5585409, -0.2185788, 0.3634375, -0.7530441, 0.7771197
4: -0.5921478, 0.5744159, -0.3378704, 0.3002744, -0.8924221, 0.9122863
5: -0.0565934, 1.1349068, 0.3919799, 1.0857860, -1.1423793, 0.7429268
6: -0.4773337, 0.6127940, -0.2806397, 0.3743095, -0.8516432, 0.8934337
7: -0.5061942, 0.5908199, -0.3069979, 0.2658585, -0.7720527, 0.8978179
8: -0.6400030, 0.7237492, -0.3633837, 0.4554020, -1.0954050, 1.0871328
9: -0.5454586, 0.6140428, -0.2641246, 0.3566399, -0.9020984, 0.8781674

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.16 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.15 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.6470444, 0.7166237, -0.4786690, 0.5162646, -1.1633090, 1.1952927
1: -0.4657074, 0.4806128, -0.3110559, 0.3299809, -0.7956883, 0.7916687
2: -0.4448388, 0.7304481, -0.3169419, 0.4668852, -0.9117240, 1.0473900
3: -0.4144266, 0.5723801, -0.2812945, 0.4555728, -0.8699993, 0.8536745
4: -0.6136307, 0.5976964, -0.4472647, 0.4222382, -1.0358689, 1.0449611
5: -0.0918896, 1.1400816, 0.2136610, 1.1092556, -1.2011452, 0.9264207
6: -0.4934038, 0.6320793, -0.3620435, 0.4841082, -0.9775119, 0.9941229
7: -0.5269539, 0.6168997, -0.3852987, 0.3958642, -0.9228181, 1.0021985
8: -0.6637725, 0.7447499, -0.4764558, 0.5743870, -1.2381594, 1.2212057
9: -0.5666962, 0.6368293, -0.3907277, 0.4525194, -1.0192156, 1.0275570

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_B1_A1_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.41 seconds

## Relational analysis of NS_B1_A1_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.33 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.2962693, 1.0802286, -0.3754404, 0.4135272, -1.7097964, 1.4556689
1: -0.8751979, 0.9223014, -0.2330824, 0.2433711, -1.1185690, 1.1553838
2: -1.0003629, 1.1816937, -0.2329484, 0.3633087, -1.3636715, 1.4146421
3: -1.1744785, 0.8105032, -0.2225529, 0.3697130, -1.5441916, 1.0330561
4: -1.1899456, 1.1013863, -0.3441457, 0.3053239, -1.4952695, 1.4455320
5: -0.8506229, 1.3055592, 0.3823985, 1.0872734, -1.9378963, 0.9231607
6: -0.9384903, 1.0872519, -0.2854034, 0.3806066, -1.3190968, 1.3726553
7: -1.0014470, 1.1334949, -0.3117994, 0.2704998, -1.2719468, 1.4452943
8: -1.2927140, 1.0873916, -0.3707477, 0.4618672, -1.7545812, 1.4581393
9: -1.0648472, 1.0722754, -0.2689949, 0.3627930, -1.4276402, 1.3412702

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.32 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.22 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.3571103, 1.1153330, -0.4893303, 0.5309633, -1.8880736, 1.6046633
1: -0.9163316, 0.9667907, -0.3221709, 0.3406204, -1.2569520, 1.2889616
2: -1.0562602, 1.2229995, -0.3262388, 0.4837413, -1.5400016, 1.5492382
3: -1.2503067, 0.8333954, -0.2894168, 0.4639458, -1.7142525, 1.1228122
4: -1.2481244, 1.1496367, -0.4577619, 0.4356082, -1.6837326, 1.6073986
5: -0.9239790, 1.3244151, 0.1917496, 1.1113803, -2.0353594, 1.1326655
6: -0.9817454, 1.1351311, -0.3712420, 0.4946001, -1.4763454, 1.5063732
7: -1.0484927, 1.1829271, -0.3947701, 0.4117590, -1.4602517, 1.5776972
8: -1.3547218, 1.1205080, -0.4903568, 0.5857323, -1.9404540, 1.6108648
9: -1.1149348, 1.1136738, -0.4038314, 0.4652639, -1.5801988, 1.5175052

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.45 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.22 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5938244, 0.6639912, -0.8026124, 0.8089141, -1.4027386, 1.4666036
1: -0.4226675, 0.4378105, -0.5561259, 0.5788398, -1.0015073, 0.9939364
2: -0.4061226, 0.6585521, -0.5637853, 0.8496674, -1.2557900, 1.2223372
3: -0.3704744, 0.5401527, -0.5784979, 0.6296074, -1.0000818, 1.1186506
4: -0.5649908, 0.5488838, -0.7352788, 0.7197605, -1.2847512, 1.2841626
5: -0.0102418, 1.1303495, -0.2658224, 1.1671066, -1.1773484, 1.3961719
6: -0.4571531, 0.5898031, -0.5946834, 0.7252014, -1.1823545, 1.1844865
7: -0.4848466, 0.5571934, -0.6384424, 0.7426727, -1.2275193, 1.1956358
8: -0.6122321, 0.6959898, -0.8003228, 0.8293414, -1.4415734, 1.4963126
9: -0.5188782, 0.5856174, -0.6757376, 0.7424628, -1.2613410, 1.2613550

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.30 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.6066934, 0.6788093, -1.5646135, 1.2337115, -1.8404049, 2.2434227
1: -0.4342327, 0.4490640, -1.0565341, 1.1168189, -1.5510516, 1.5055981
2: -0.4151252, 0.6785562, -1.2463992, 1.3622922, -1.7774174, 1.9249554
3: -0.3794613, 0.5492116, -1.5076575, 0.9105923, -1.2900536, 2.0568690
4: -0.5780947, 0.5614421, -1.4503899, 1.3123502, -1.8904449, 2.0118320
5: -0.0330769, 1.1325519, -1.1753539, 1.3880020, -1.4210789, 2.3079057
6: -0.4669446, 0.6011295, -1.1359603, 1.2965903, -1.7635349, 1.7370899
7: -0.4949140, 0.5737597, -1.2071414, 1.3496850, -1.8445989, 1.7809011
8: -0.6257544, 0.7096656, -1.5665500, 1.2321835, -1.8579378, 2.2762156
9: -0.5319730, 0.5992738, -1.2838403, 1.2540894, -1.7860624, 1.8831141

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.32 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.17 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.8026124, 0.8089141, -0.9878203, 0.9037950, -1.7064074, 1.7967345
1: -0.5561259, 0.5788398, -0.6668415, 0.6992025, -1.2553284, 1.2456813
2: -0.5637853, 0.8496674, -0.7183757, 0.9723207, -1.5361059, 1.5680430
3: -0.5784979, 0.6296074, -0.7913158, 0.6954112, -1.2739091, 1.4209232
4: -0.7352788, 0.7197605, -0.8961803, 0.8581489, -1.5934278, 1.6159408
5: -0.2658224, 1.1671066, -0.4793717, 1.2113708, -1.4771932, 1.6464783
6: -0.5946834, 0.7252014, -0.7200980, 0.8466926, -1.4413760, 1.4452994
7: -0.6384424, 0.7426727, -0.7651681, 0.8840750, -1.5225174, 1.5078409
8: -0.8003228, 0.8293414, -0.9790577, 0.9203307, -1.7206535, 1.8083991
9: -0.6757376, 0.7424628, -0.8119997, 0.8632962, -1.5390339, 1.5544624

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.26 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.13 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.5646135, 1.2337115, -1.0224509, 0.9226260, -2.4872394, 2.2561624
1: -1.0565341, 1.1168189, -0.6900517, 0.7231234, -1.7796575, 1.8068707
2: -1.2463992, 1.3622922, -0.7493122, 0.9957635, -2.2421627, 2.1116042
3: -1.5076575, 0.9105923, -0.8337490, 0.7078515, -2.2155089, 1.7443413
4: -1.4503899, 1.3123502, -0.9284760, 0.8844820, -2.3348718, 2.2408261
5: -1.1753539, 1.3880020, -0.5206037, 1.2211555, -2.3965094, 1.9086057
6: -1.1359603, 1.2965903, -0.7439749, 0.8728732, -2.0088336, 2.0405653
7: -1.2071414, 1.3496850, -0.7905702, 0.9111913, -2.1183326, 2.1402552
8: -1.5665500, 1.2321835, -1.0137864, 0.9388161, -2.5053661, 2.2459698
9: -1.2838403, 1.2540894, -0.8398165, 0.8864326, -2.1702728, 2.0939059

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.30 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.51 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4284188, 0.4649323, -0.3209792, 0.3489072, -0.7773260, 0.7859116
1: -0.2720387, 0.2815433, -0.1935983, 0.2055060, -0.4775447, 0.4751415
2: -0.2730199, 0.4178401, -0.1932847, 0.3014991, -0.5745190, 0.6111248
3: -0.2524263, 0.4150866, -0.1860236, 0.3131845, -0.5656108, 0.6011102
4: -0.3986038, 0.3598342, -0.2907199, 0.2589112, -0.6575150, 0.6505541
5: 0.3007306, 1.0989044, 0.4704674, 1.0772523, -0.7765217, 0.6284369
6: -0.3233039, 0.4338188, -0.2426971, 0.3227254, -0.6460292, 0.6765159
7: -0.3477758, 0.3304052, -0.2686645, 0.2308226, -0.5785984, 0.5990697
8: -0.4240641, 0.5193204, -0.3069575, 0.4025055, -0.8265696, 0.8262780
9: -0.3276704, 0.4068694, -0.2262427, 0.3062355, -0.6339059, 0.6331121

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8013404, upper bound: 1.7950476
time: 1.56 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
time: 1.35 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5064518, 0.5549020, -0.3447450, 0.3770972, -0.8835490, 0.8996471
1: -0.3391015, 0.3576271, -0.2109341, 0.2220588, -0.5611603, 0.5685613
2: -0.3403986, 0.5146055, -0.2103595, 0.3280855, -0.6684840, 0.7249650
3: -0.3041497, 0.4768026, -0.2020898, 0.3377175, -0.6418672, 0.6788924
4: -0.4751630, 0.4561970, -0.3130799, 0.2793242, -0.7544872, 0.7692769
5: 0.1563377, 1.1146208, 0.4317333, 1.0805568, -0.9242191, 0.6828876
6: -0.3861614, 0.5108414, -0.2610384, 0.3481826, -0.7343439, 0.7718797
7: -0.4108629, 0.4373840, -0.2871882, 0.2476684, -0.6585313, 0.7245722
8: -0.5124327, 0.6030131, -0.3338551, 0.4286104, -0.9410431, 0.9368682
9: -0.4240521, 0.4861137, -0.2445538, 0.3311106, -0.7551627, 0.7306675

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.8018666
time: 1.63 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
time: 1.46 seconds

## BFS NS instance: NS_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.5555204, 0.6198856, -0.3476999, 0.3806020, -0.9361224, 0.9675854
1: -0.3882424, 0.4043145, -0.2130895, 0.2241170, -0.6123593, 0.6174040
2: -0.3793257, 0.5990111, -0.2124825, 0.3313911, -0.7107167, 0.8114935
3: -0.3437257, 0.5131882, -0.2040873, 0.3407677, -0.6844934, 0.7172755
4: -0.5259879, 0.5115029, -0.3158599, 0.2818623, -0.8078502, 0.8273627
5: 0.0577280, 1.1237942, 0.4269174, 1.0809677, -1.0232397, 0.6968768
6: -0.4280085, 0.5560894, -0.2633188, 0.3513476, -0.7793561, 0.8194082
7: -0.4548810, 0.5078845, -0.2894911, 0.2497627, -0.7046437, 0.7973756
8: -0.5719828, 0.6552828, -0.3371992, 0.4318563, -1.0038391, 0.9924820
9: -0.4799010, 0.5449692, -0.2468306, 0.3342032, -0.8141042, 0.7917998

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B2_B1_B1

### Relational analysis result of NS_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.8018571
time: 1.43 seconds

## Relational analysis of NS_B1_A1_B2_B1_B2

### Relational analysis result of NS_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
time: 1.43 seconds

## BFS NS instance: NS_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.5927643, 0.6627703, -0.3963786, 0.4343500, -1.0271142, 1.0591489
1: -0.4217145, 0.4368833, -0.2468828, 0.2582774, -0.6799919, 0.6837661
2: -0.4053806, 0.6569039, -0.2468674, 0.3856780, -0.7910586, 0.9037713
3: -0.3697341, 0.5394062, -0.2344479, 0.3883056, -0.7580397, 0.7738540
4: -0.5639116, 0.5478486, -0.3663573, 0.3221762, -0.8860878, 0.9142059
5: -0.0083600, 1.1301681, 0.3523340, 1.0920444, -1.1004044, 0.7778341
6: -0.4563464, 0.5888698, -0.3007571, 0.4007336, -0.8570800, 0.8896269
7: -0.4840170, 0.5558288, -0.3259081, 0.2903583, -0.7743753, 0.8817369
8: -0.6111180, 0.6948627, -0.3923866, 0.4835427, -1.0946608, 1.0872493
9: -0.5177991, 0.5844923, -0.2891463, 0.3808738, -0.8986729, 0.8736386

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B2_B2_B1

### Relational analysis result of NS_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.8018563
time: 1.58 seconds

## Relational analysis of NS_B1_A1_B2_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
time: 1.42 seconds

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.6706032, 0.7316090, -0.3268338, 0.3558514, -1.0264547, 1.0584428
1: -0.4798984, 0.4961185, -0.1978688, 0.2095836, -0.6894820, 0.6939873
2: -0.4620208, 0.7515420, -0.1974908, 0.3080485, -0.7700694, 0.9490328
3: -0.4389873, 0.5820788, -0.1899813, 0.3192281, -0.7582154, 0.7720600
4: -0.6308182, 0.6167731, -0.2962283, 0.2639399, -0.8947581, 0.9130013
5: -0.1178967, 1.1444142, 0.4609255, 1.0780662, -1.1959629, 0.6834887
6: -0.5065662, 0.6466463, -0.2472154, 0.3289964, -0.8355626, 0.8938617
7: -0.5450510, 0.6365256, -0.2732277, 0.2349724, -0.7800235, 0.9097533
8: -0.6832774, 0.7596160, -0.3135834, 0.4089362, -1.0922135, 1.0731995
9: -0.5842836, 0.6534452, -0.2307535, 0.3123634, -0.8966470, 0.8841988

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8013227, upper bound: 1.7950476
time: 1.62 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.67 seconds

## BFS NS instance: NS_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9048848, 0.8608105, -0.3505303, 0.3840135, -1.2888982, 1.2113408
1: -0.6162655, 0.6422341, -0.2151541, 0.2261109, -0.8423765, 0.8573883
2: -0.6478100, 0.9160859, -0.2146026, 0.3345574, -0.9823674, 1.1306885
3: -0.6925324, 0.6656105, -0.2060008, 0.3437297, -1.0362620, 0.8716112
4: -0.8211428, 0.7964522, -0.3186458, 0.2842933, -1.1054361, 1.1150980
5: -0.3824525, 1.1879985, 0.4223042, 1.0815150, -1.4639676, 0.7656943
6: -0.6631646, 0.7888780, -0.2656020, 0.3543794, -1.0175439, 1.0544800
7: -0.7075459, 0.8195910, -0.2918019, 0.2517692, -0.9593151, 1.1113930
8: -0.8972346, 0.8773325, -0.3405572, 0.4349654, -1.3322001, 1.2178897
9: -0.7472359, 0.8080876, -0.2490527, 0.3371661, -1.0844020, 1.0571404

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8013105, upper bound: 1.7950476
time: 1.63 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.23 seconds

## BFS NS instance: NS_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1.0865023, 0.9591967, -0.3554177, 0.3899045, -1.4764068, 1.3146144
1: -0.7333767, 0.7689120, -0.2187191, 0.2295544, -0.9629310, 0.9876310
2: -0.8076408, 1.0392793, -0.2182644, 0.3400251, -1.1476659, 1.2575437
3: -0.9130412, 0.7315766, -0.2093046, 0.3488445, -1.2618858, 0.9408812
4: -0.9893591, 0.9350271, -0.3234577, 0.2884911, -1.2778503, 1.2584848
5: -0.5977082, 1.2405468, 0.4143391, 1.0824621, -1.6801703, 0.8262078
6: -0.7893558, 0.9221752, -0.2695458, 0.3596144, -1.1489701, 1.1917210
7: -0.8392434, 0.9630637, -0.2957935, 0.2552333, -1.0944767, 1.2588573
8: -1.0789256, 0.9732134, -0.3463573, 0.4403334, -1.5192590, 1.3195707
9: -0.8921571, 0.9295425, -0.2528902, 0.3422810, -1.2344382, 1.1824328

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8012567, upper bound: 1.7950476
time: 1.60 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.35 seconds

## BFS NS instance: NS_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1.2060897, 1.0281971, -0.4061598, 0.4435579, -1.6496477, 1.4343568
1: -0.8142285, 0.8563591, -0.2544653, 0.2650220, -1.0792505, 1.1108245
2: -0.9175107, 1.1204699, -0.2545951, 0.3955703, -1.3130810, 1.3750651
3: -1.0620854, 0.7765726, -0.2399585, 0.3965395, -1.4586248, 1.0165311
4: -1.1037130, 1.0298672, -0.3762755, 0.3331810, -1.4368941, 1.4061427
5: -0.7418943, 1.2776101, 0.3367803, 1.0941542, -1.8360486, 0.9408298
6: -0.8743767, 1.0162848, -0.3075467, 0.4107426, -1.2851193, 1.3238316
7: -0.9317151, 1.0602258, -0.3323733, 0.3026759, -1.2343910, 1.3925991
8: -1.2008058, 1.0383061, -0.4019556, 0.4943068, -1.6951126, 1.4402617
9: -0.9906073, 1.0109144, -0.3005819, 0.3888692, -1.3794764, 1.3114964

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8013228, upper bound: 1.7950476
time: 1.75 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.34 seconds

## BFS NS instance: NS_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3694080, 0.4064409, -0.6307308, 0.7042207, -1.0736287, 1.0371717
1: -0.2287708, 0.2392205, -0.4549764, 0.4687619, -0.6975328, 0.6941968
2: -0.2285435, 0.3562610, -0.4327844, 0.7128482, -0.9413916, 0.7890454
3: -0.2185788, 0.3634375, -0.3977718, 0.5647414, -0.7833201, 0.7612093
4: -0.3378704, 0.3002744, -0.6016312, 0.5837743, -0.9216447, 0.9019055
5: 0.3919799, 1.0857860, -0.0722234, 1.1367196, -0.7447397, 1.1580093
6: -0.2806397, 0.3743095, -0.4842878, 0.6208997, -0.9015394, 0.8585973
7: -0.3069979, 0.2658585, -0.5145020, 0.6021583, -0.9091562, 0.7803605
8: -0.3633837, 0.4554020, -0.6497869, 0.7331098, -1.0964935, 1.1051890
9: -0.2641246, 0.3566399, -0.5544902, 0.6239954, -0.8881200, 0.9111301

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_B2_A1_B1_A1_B1

### Relational analysis result of NS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.31 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2

### Relational analysis result of NS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.20 seconds

## BFS NS instance: NS_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4786690, 0.5162646, -0.6643344, 0.7277970, -1.2064661, 1.1805990
1: -0.3110559, 0.3299809, -0.4763325, 0.4921241, -0.8031800, 0.8063133
2: -0.3169419, 0.4668852, -0.4574634, 0.7462483, -1.0631902, 0.9243487
3: -0.2812945, 0.4555728, -0.4325648, 0.5796445, -0.8609390, 0.8881376
4: -0.4472647, 0.4222382, -0.6263273, 0.6119225, -1.0591872, 1.0485655
5: 0.2136610, 1.1092556, -0.1112646, 1.1432925, -0.9296316, 1.2205201
6: -0.3620435, 0.4841082, -0.5031152, 0.6428936, -1.0049372, 0.9872233
7: -0.3852987, 0.3958642, -0.5403631, 0.6314607, -1.0167594, 0.9362272
8: -0.4764558, 0.5743870, -0.6782587, 0.7558283, -1.2322841, 1.2526456
9: -0.3907277, 0.4525194, -0.5797680, 0.6491210, -1.0398488, 1.0322874

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of NS_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.37 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.35 seconds

## BFS NS instance: NS_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3754404, 0.4135272, -1.3200198, 1.0933015, -1.4687419, 1.7335470
1: -0.2330824, 0.2433711, -0.8912140, 0.9388695, -1.1719519, 1.1345850
2: -0.2329484, 0.3633087, -1.0219481, 1.1970766, -1.4300249, 1.3852568
3: -0.2225529, 0.3697130, -1.2034862, 0.8190284, -1.0415814, 1.5731993
4: -0.3441457, 0.3053239, -1.2144588, 1.1193542, -1.4634999, 1.5197828
5: 0.3823985, 1.0872734, -0.8798171, 1.3125809, -0.9301825, 1.9670905
6: -0.2854034, 0.3806066, -0.9585111, 1.1050825, -1.3904859, 1.3391176
7: -0.3117994, 0.2704998, -1.0189674, 1.1519318, -1.4637312, 1.2894672
8: -0.3707477, 0.4618672, -1.3170832, 1.0997239, -1.4704716, 1.7789505
9: -0.2689949, 0.3627930, -1.0835001, 1.0880721, -1.3570669, 1.4462931

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.35 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.26 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4893303, 0.5309633, -1.3827219, 1.1292959, -1.6186262, 1.9136852
1: -0.3221709, 0.3406204, -0.9335941, 0.9844870, -1.3066579, 1.2742145
2: -0.3262388, 0.4837413, -1.0794863, 1.2394298, -1.5656686, 1.5632277
3: -0.2894168, 0.4639458, -1.2814620, 0.8425009, -1.1319177, 1.7454078
4: -0.4577619, 0.4356082, -1.2749401, 1.1688296, -1.6265914, 1.7105484
5: 0.1917496, 1.1113803, -0.9555786, 1.3319154, -1.1401658, 2.0669589
6: -0.3712420, 0.4946001, -1.0040007, 1.1541759, -1.5254179, 1.4986007
7: -0.3947701, 0.4117590, -1.0672060, 1.2026267, -1.5973967, 1.4789650
8: -0.4903568, 0.5857323, -1.3810341, 1.1336806, -1.6240374, 1.9667664
9: -0.4038314, 0.4652639, -1.1348579, 1.1306312, -1.5344626, 1.6001217

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of NS_B2_A1_B2_A2_A1

### Relational analysis result of NS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.41 seconds

## Relational analysis of NS_B2_A1_B2_A2_A2

### Relational analysis result of NS_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.26 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.8026124, 0.8089141, -0.8026124, 0.8089141, -1.6115265, 1.6115265
1: -0.5561259, 0.5788398, -0.5561259, 0.5788398, -1.1349657, 1.1349657
2: -0.5637853, 0.8496674, -0.5637853, 0.8496674, -1.4134526, 1.4134526
3: -0.5784979, 0.6296074, -0.5784979, 0.6296074, -1.2081053, 1.2081053
4: -0.7352788, 0.7197605, -0.7352788, 0.7197605, -1.4550393, 1.4550393
5: -0.2658224, 1.1671066, -0.2658224, 1.1671066, -1.4329290, 1.4329290
6: -0.5946834, 0.7252014, -0.5946834, 0.7252014, -1.3198848, 1.3198848
7: -0.6384424, 0.7426727, -0.6384424, 0.7426727, -1.3811152, 1.3811152
8: -0.8003228, 0.8293414, -0.8003228, 0.8293414, -1.6296642, 1.6296642
9: -0.6757376, 0.7424628, -0.6757376, 0.7424628, -1.4182005, 1.4182005

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of NS_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.47 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.18 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.8026124, 0.8089141, -1.5600469, 1.2149606, -2.0175729, 2.3689611
1: -0.5561259, 0.5788398, -1.0503581, 1.0944669, -1.6505928, 1.6291978
2: -0.5637853, 0.8496674, -1.2295642, 1.3596848, -1.9234700, 2.0792315
3: -0.5784979, 0.6296074, -1.4924760, 0.9009742, -1.4794720, 2.1220834
4: -0.7352788, 0.7197605, -1.4298247, 1.2932727, -2.0285516, 2.1495852
5: -0.2658224, 1.1671066, -1.1606829, 1.3730503, -1.6388727, 2.3277895
6: -0.5946834, 0.7252014, -1.1146259, 1.2792917, -1.8739752, 1.8398273
7: -0.6384424, 0.7426727, -1.1849133, 1.3321381, -1.9705805, 1.9275861
8: -0.8003228, 0.8293414, -1.5528996, 1.2257862, -2.0261090, 2.3822410
9: -0.6757376, 0.7424628, -1.2716372, 1.2455933, -1.9213309, 2.0141001

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of NS_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.43 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.30 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1.5646135, 1.2337115, -0.8026124, 0.8089141, -2.3735275, 2.0363240
1: -1.0565341, 1.1168189, -0.5561259, 0.5788398, -1.6353738, 1.6729448
2: -1.2463992, 1.3622922, -0.5637853, 0.8496674, -2.0960665, 1.9260774
3: -1.5076575, 0.9105923, -0.5784979, 0.6296074, -2.1372650, 1.4890902
4: -1.4503899, 1.3123502, -0.7352788, 0.7197605, -2.1701503, 2.0476289
5: -1.1753539, 1.3880020, -0.2658224, 1.1671066, -2.3424606, 1.6538244
6: -1.1359603, 1.2965903, -0.5946834, 0.7252014, -1.8611617, 1.8912737
7: -1.2071414, 1.3496850, -0.6384424, 0.7426727, -1.9498141, 1.9881274
8: -1.5665500, 1.2321835, -0.8003228, 0.8293414, -2.3958914, 2.0325062
9: -1.2838403, 1.2540894, -0.6757376, 0.7424628, -2.0263031, 1.9298270

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.34 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.32 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1.5646135, 1.2337115, -1.5600469, 1.2149606, -2.7795739, 2.7937584
1: -1.0565341, 1.1168189, -1.0503581, 1.0944669, -2.1510010, 2.1671770
2: -1.2463992, 1.3622922, -1.2295642, 1.3596848, -2.6060839, 2.5918565
3: -1.5076575, 0.9105923, -1.4924760, 0.9009742, -2.4086318, 2.4030683
4: -1.4503899, 1.3123502, -1.4298247, 1.2932727, -2.7436626, 2.7421749
5: -1.1753539, 1.3880020, -1.1606829, 1.3730503, -2.5484042, 2.5486851
6: -1.1359603, 1.2965903, -1.1146259, 1.2792917, -2.4152522, 2.4112163
7: -1.2071414, 1.3496850, -1.1849133, 1.3321381, -2.5392795, 2.5345984
8: -1.5665500, 1.2321835, -1.5528996, 1.2257862, -2.7923362, 2.7850831
9: -1.2838403, 1.2540894, -1.2716372, 1.2455933, -2.5294337, 2.5257266

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of NS_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.48 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.35 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.79 seconds
NS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8013404, upper bound: 1.7950476
NS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
NS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7951458, upper bound: 1.8018666
NS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
NS_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7951458, upper bound: 1.8018571
NS_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
NS_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7951458, upper bound: 1.8018563
NS_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
NS_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8013227, upper bound: 1.7950476
NS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8013105, upper bound: 1.7950476
NS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8012567, upper bound: 1.7950476
NS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8013228, upper bound: 1.7950476
NS_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

## BFS NS instance: NS_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.3657324, 0.4021144, -0.3068123, 0.3321023, -0.6978346, 0.7089267
1: -0.2261384, 0.2366914, -0.1832643, 0.1956385, -0.4217769, 0.4199557
2: -0.2258541, 0.3519635, -0.1831059, 0.2856508, -0.5115049, 0.5350693
3: -0.2161521, 0.3596064, -0.1764463, 0.2985602, -0.5147123, 0.5360528
4: -0.3340492, 0.2971913, -0.2773910, 0.2467429, -0.5807921, 0.5745823
5: 0.3978301, 1.0848793, 0.4935573, 1.0752822, -0.6774521, 0.5913219
6: -0.2777357, 0.3704647, -0.2317638, 0.3075498, -0.5852855, 0.6022284
7: -0.3040664, 0.2630284, -0.2576224, 0.2207806, -0.5248470, 0.5206509
8: -0.3588872, 0.4514598, -0.2909233, 0.3869441, -0.7458314, 0.7423830
9: -0.2611597, 0.3528830, -0.2153271, 0.2914075, -0.5525671, 0.5682100

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A1_B1_A1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7814949, upper bound: 1.7511661
time: 1.49 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7679741, upper bound: 1.7511661
time: 1.35 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.6068220, 0.6515751, -0.2852308, 0.3060575, -0.9128795, 0.9368059
1: -0.4128872, 0.4457426, -0.1672477, 0.1803475, -0.5932347, 0.6129904
2: -0.4211792, 0.5963285, -0.1673304, 0.2616305, -0.6828097, 0.7636589
3: -0.3523533, 0.5639850, -0.1617797, 0.2758944, -0.6282476, 0.7257648
4: -0.5775616, 0.5740696, -0.2568123, 0.2278831, -0.8054447, 0.8308818
5: 0.0045716, 1.1369739, 0.5293438, 1.0723630, -1.0677915, 0.6076300
6: -0.4608154, 0.6187713, -0.2151716, 0.2840299, -0.7448453, 0.8339429
7: -0.4857764, 0.5526510, -0.2405086, 0.2057653, -0.6915417, 0.7931597
8: -0.6147203, 0.7198025, -0.2661728, 0.3628254, -0.9775457, 0.9859753
9: -0.5595804, 0.5511364, -0.1984092, 0.2685689, -0.8281493, 0.7495456

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A1_B1_A1_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7740677, upper bound: 1.7511661
time: 1.52 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7511661
time: 1.33 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4858801, 0.5260668, -0.2930965, 0.3156731, -0.8015532, 0.8191634
1: -0.3187082, 0.3371679, -0.1731610, 0.1859921, -0.5047003, 0.5103289
2: -0.3233421, 0.4774862, -0.1731547, 0.2703509, -0.5936930, 0.6506408
3: -0.2865390, 0.4613173, -0.1671464, 0.2842625, -0.5708016, 0.6284637
4: -0.4542625, 0.4313959, -0.2643881, 0.2348459, -0.6891084, 0.6957841
5: 0.1989636, 1.1107183, 0.5161319, 1.0734043, -0.8744407, 0.5945864
6: -0.3682292, 0.4912775, -0.2212012, 0.2927135, -0.6609427, 0.7124788
7: -0.3914780, 0.4065697, -0.2468269, 0.2111600, -0.6026379, 0.6533966
8: -0.4858544, 0.5821977, -0.2752834, 0.3717296, -0.8575841, 0.8574811
9: -0.3996980, 0.4610791, -0.2046551, 0.2769617, -0.6766598, 0.6657342

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7843712
time: 1.31 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7703235
time: 1.26 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4521858, 0.4892803, -0.4816105, 0.5394434, -0.9916292, 0.9708908
1: -0.2904126, 0.3040384, -0.3107700, 0.3173861, -0.6077987, 0.6148084
2: -0.2933790, 0.4411240, -0.3086933, 0.4811963, -0.7745754, 0.7498173
3: -0.2658479, 0.4345105, -0.2946143, 0.4790012, -0.7448491, 0.7291248
4: -0.4219492, 0.3889593, -0.4418494, 0.3968819, -0.8188311, 0.8308088
5: 0.2603597, 1.1038705, 0.2086653, 1.0995879, -0.8392282, 0.8952052
6: -0.3415779, 0.4579461, -0.3666645, 0.4947889, -0.8363668, 0.8246107
7: -0.3657780, 0.3607229, -0.3938640, 0.3446821, -0.7104601, 0.7545869
8: -0.4489353, 0.5456309, -0.4887579, 0.5789480, -1.0278833, 1.0343889
9: -0.3579234, 0.4273470, -0.3500077, 0.4743637, -0.8322871, 0.7773547

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B1_A1_B1_A2_B2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7763739
time: 1.23 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7511661
time: 1.18 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.5286555, 0.5856364, -0.2931429, 0.3157300, -0.8443855, 0.8787793
1: -0.3616810, 0.3793340, -0.1731959, 0.1860255, -0.5477066, 0.5525299
2: -0.3586897, 0.5540992, -0.1731890, 0.2704023, -0.6290920, 0.7272882
3: -0.3229548, 0.4934987, -0.1671781, 0.2843120, -0.6072668, 0.6606768
4: -0.4980999, 0.4824762, -0.2644329, 0.2348872, -0.7329871, 0.7469091
5: 0.1102437, 1.1188924, 0.5160534, 1.0734106, -0.9631670, 0.6028390
6: -0.4056692, 0.5315722, -0.2212371, 0.2927647, -0.6984340, 0.7528093
7: -0.4316125, 0.4705589, -0.2468642, 0.2111918, -0.6428043, 0.7174231
8: -0.5407287, 0.6261013, -0.2753370, 0.3717824, -0.9125111, 0.9014383
9: -0.4502872, 0.5134728, -0.2046920, 0.2770114, -0.7272986, 0.7181648

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B1_A1_B2_B1_B1_B1

### Relational analysis result of NS_B1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7843712
time: 1.36 seconds

## Relational analysis of NS_B1_A1_B2_B1_B1_B2

### Relational analysis result of NS_B1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7702354
time: 1.29 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.4917094, 0.5342917, -0.4616129, 0.5157229, -1.0074322, 0.9959046
1: -0.3245249, 0.3429850, -0.2961828, 0.3034576, -0.6279825, 0.6391678
2: -0.3282074, 0.4880323, -0.2943257, 0.4588248, -0.7870322, 0.7823581
3: -0.2914653, 0.4657325, -0.2810954, 0.4583581, -0.7498235, 0.7468280
4: -0.4601792, 0.4384705, -0.4230347, 0.3797054, -0.8398846, 0.8615052
5: 0.1868260, 1.1118304, 0.2412581, 1.0968072, -0.9099813, 0.8705723
6: -0.3733162, 0.4968581, -0.3512313, 0.4733684, -0.8466846, 0.8480893
7: -0.3970074, 0.4153199, -0.3782773, 0.3305074, -0.7275148, 0.7935972
8: -0.4934260, 0.5881350, -0.4661249, 0.5569820, -1.0504080, 1.0542599
9: -0.4066411, 0.4681631, -0.3345997, 0.4534325, -0.8600736, 0.8027628

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B1_A1_B2_B1_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7763739
time: 1.24 seconds

## Relational analysis of NS_B1_A1_B2_B1_B2_B2

### Relational analysis result of NS_B1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7511661
time: 1.23 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.5628905, 0.6283717, -0.3390830, 0.3703811, -0.9332716, 0.9674547
1: -0.3948663, 0.4107593, -0.2068040, 0.2181151, -0.6129814, 0.6175632
2: -0.3844815, 0.6104673, -0.2062916, 0.3217516, -0.7062331, 0.8167588
3: -0.3488725, 0.5183766, -0.1982621, 0.3318728, -0.6807452, 0.7166387
4: -0.5334927, 0.5186954, -0.3077528, 0.2744609, -0.8079536, 0.8264481
5: 0.0446496, 1.1250556, 0.4409614, 1.0797694, -1.0351198, 0.6840941
6: -0.4336163, 0.5625763, -0.2566687, 0.3421173, -0.7757336, 0.8192450
7: -0.4606466, 0.5173721, -0.2827750, 0.2436551, -0.7043016, 0.8001471
8: -0.5797271, 0.6631154, -0.3274469, 0.4223912, -1.0021183, 0.9905623
9: -0.4874005, 0.5527905, -0.2401915, 0.3251842, -0.8125846, 0.7929819

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B1_A1_B2_B2_B1_B1

### Relational analysis result of NS_B1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7843712
time: 1.29 seconds

## Relational analysis of NS_B1_A1_B2_B2_B1_B2

### Relational analysis result of NS_B1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7702354
time: 1.22 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.5225610, 0.5771261, -0.5677752, 0.5995942, -1.1221552, 1.1449013
1: -0.3550973, 0.3733763, -0.3742709, 0.3793174, -0.7344147, 0.7476473
2: -0.3535787, 0.5432601, -0.3681867, 0.5631977, -0.9167765, 0.9114468
3: -0.3177934, 0.4888853, -0.3322715, 0.5360665, -0.8538600, 0.8211567
4: -0.4917163, 0.4752635, -0.5443439, 0.4877368, -0.9794530, 1.0196074
5: 0.1232671, 1.1177198, 0.0858025, 1.1299075, -1.0066403, 1.0319173
6: -0.4001473, 0.5258820, -0.4226010, 0.5709791, -0.9711263, 0.9484830
7: -0.4258308, 0.4614533, -0.4378710, 0.5113983, -0.9372292, 0.8993243
8: -0.5329626, 0.6193352, -0.5641079, 0.6632349, -1.1961975, 1.1834431
9: -0.4430864, 0.5056530, -0.4819160, 0.5243580, -0.9674444, 0.9875690

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B1_A1_B2_B2_B2_B1

### Relational analysis result of NS_B1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7763739
time: 1.32 seconds

## Relational analysis of NS_B1_A1_B2_B2_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7511661
time: 1.20 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.5534362, 0.6174856, -0.3126657, 0.3390453, -0.8924815, 0.9301513
1: -0.3863693, 0.4024916, -0.1875339, 0.1997154, -0.5860847, 0.5900255
2: -0.3778673, 0.5957717, -0.1873114, 0.2921986, -0.6700659, 0.7830831
3: -0.3422702, 0.5117209, -0.1804033, 0.3046026, -0.6468728, 0.6921241
4: -0.5238657, 0.5094687, -0.2828979, 0.2517706, -0.7756363, 0.7923666
5: 0.0614264, 1.1234374, 0.4840174, 1.0760959, -1.0146695, 0.6394200
6: -0.4264226, 0.5542547, -0.2362811, 0.3138198, -0.7402424, 0.7905357
7: -0.4532507, 0.5052010, -0.2621846, 0.2249296, -0.6781803, 0.7673855
8: -0.5697927, 0.6530678, -0.2975479, 0.3933736, -0.9631664, 0.9506156
9: -0.4777798, 0.5427573, -0.2198371, 0.2975337, -0.7753135, 0.7625944

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B1_A1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7813746, upper bound: 1.7511661
time: 1.57 seconds

## Relational analysis of NS_B1_A2_B1_A1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7678785, upper bound: 1.7511661
time: 1.39 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1.1756120, 1.0650010, -0.2908989, 0.3129864, -1.4885983, 1.3558999
1: -0.7684968, 0.8179273, -0.1715090, 0.1844152, -0.9529120, 0.9894363
2: -0.8442120, 1.1780320, -0.1715274, 0.2679143, -1.1121264, 1.3495594
3: -0.9615310, 0.7781972, -0.1656470, 0.2819246, -1.2434556, 0.9438442
4: -1.0177187, 1.0075417, -0.2622715, 0.2329008, -1.2506194, 1.2698132
5: -0.6614050, 1.2347685, 0.5198231, 1.0731133, -1.7345183, 0.7149454
6: -0.7988299, 0.9489495, -0.2195167, 0.2902872, -1.0891171, 1.1684662
7: -0.9450431, 1.0445442, -0.2450617, 0.2096527, -1.1546957, 1.2896060
8: -1.0977079, 1.0647178, -0.2727380, 0.3692421, -1.4669499, 1.3374557
9: -0.9656072, 1.0017703, -0.2029101, 0.2746170, -1.2402241, 1.2046803

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B1_A1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7738816, upper bound: 1.7511661
time: 1.42 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7511661
time: 1.24 seconds

## BFS NS instance: NS_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.6497580, 0.7185260, -0.3362445, 0.3670138, -1.0167718, 1.0547705
1: -0.4675128, 0.4824693, -0.2047334, 0.2161381, -0.6836509, 0.6872027
2: -0.4468308, 0.7331409, -0.2042521, 0.3185761, -0.7654069, 0.9373930
3: -0.4172420, 0.5736172, -0.1963432, 0.3289424, -0.7461845, 0.7699604
4: -0.6156726, 0.6000744, -0.3050821, 0.2720228, -0.8876954, 0.9051565
5: -0.0950567, 1.1406139, 0.4455880, 1.0793748, -1.1744314, 0.6950259
6: -0.4948562, 0.6338754, -0.2544780, 0.3390768, -0.8339330, 0.8883534
7: -0.5290979, 0.6192842, -0.2805625, 0.2416429, -0.7707407, 0.8998466
8: -0.6660928, 0.7466088, -0.3242342, 0.4192732, -1.0853660, 1.0708430
9: -0.5688784, 0.6388442, -0.2380043, 0.3222132, -0.8910916, 0.8768485

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7813746, upper bound: 1.7511661
time: 1.61 seconds

## Relational analysis of NS_B1_A2_B1_A2_A1_A2

### Relational analysis result of NS_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7678785, upper bound: 1.7511661
time: 1.56 seconds

## BFS NS instance: NS_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1.7524234, 1.2930143, -0.3144281, 0.3411362, -2.0935595, 1.6074424
1: -1.1198447, 1.2218468, -0.1888195, 0.2009431, -1.3207878, 1.4106663
2: -1.3524746, 1.4908941, -0.1885778, 0.2941701, -1.6466446, 1.6794719
3: -1.6824850, 0.9698275, -0.1815948, 0.3064219, -1.9889069, 1.1514223
4: -1.5629408, 1.4263796, -0.2845563, 0.2532840, -1.8162248, 1.7109358
5: -1.3551323, 1.4267993, 0.4811449, 1.0763413, -2.4314737, 0.9456544
6: -1.2419311, 1.3586328, -0.2376412, 0.3157079, -1.5576390, 1.5962739
7: -1.2859473, 1.4763823, -0.2635584, 0.2261789, -1.5121262, 1.7399406
8: -1.7088894, 1.3155767, -0.2995428, 0.3953093, -2.1041987, 1.6151195
9: -1.3915401, 1.3710343, -0.2211950, 0.2993786, -1.6909187, 1.5922292

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A2_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7861047, upper bound: 1.7903640
time: 1.49 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
time: 1.38 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.7668940, 0.7898654, -0.3383390, 0.3694990, -1.1363930, 1.1282043
1: -0.5353134, 0.5570607, -0.2062614, 0.2175972, -0.7529106, 0.7633221
2: -0.5357834, 0.8251573, -0.2057571, 0.3209193, -0.8567027, 1.0309144
3: -0.5397528, 0.6177592, -0.1977593, 0.3311050, -0.8708577, 0.8155185
4: -0.7066991, 0.6919516, -0.3070530, 0.2738221, -0.9805212, 0.9990046
5: -0.2261708, 1.1608790, 0.4421737, 1.0796660, -1.3058368, 0.7187053
6: -0.5705451, 0.7040707, -0.2560947, 0.3413207, -0.9118658, 0.9601655
7: -0.6139738, 0.7154873, -0.2821952, 0.2431276, -0.8571014, 0.9976825
8: -0.7679390, 0.8119076, -0.3266051, 0.4215740, -1.1895130, 1.1385126
9: -0.6523275, 0.7192434, -0.2396183, 0.3244056, -0.9767331, 0.9588617

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7812601, upper bound: 1.7511661
time: 1.57 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7675506, upper bound: 1.7511661
time: 1.40 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1.9222194, 1.4413887, -0.3138034, 0.3403951, -2.2626145, 1.7551922
1: -1.2983944, 1.3800195, -0.1883638, 0.2005080, -1.4989024, 1.5683833
2: -1.5754504, 1.6066599, -0.1881289, 0.2934714, -1.8689218, 1.7947888
3: -1.9546157, 1.0460224, -0.1811725, 0.3057770, -2.2603927, 1.2271949
4: -1.7885003, 1.5978050, -0.2839686, 0.2527475, -2.0412478, 1.8817736
5: -1.6053267, 1.4995562, 0.4821630, 1.0762542, -2.6815810, 1.0173931
6: -1.3835108, 1.5798454, -0.2371591, 0.3150386, -1.6985494, 1.8170046
7: -1.4854664, 1.6420643, -0.2630714, 0.2257362, -1.7112025, 1.9051358
8: -1.9306645, 1.4281018, -0.2988358, 0.3946233, -2.3252878, 1.7269375
9: -1.5801599, 1.4981928, -0.2207136, 0.2987248, -1.8788848, 1.7189064

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7861047, upper bound: 1.7903640
time: 1.38 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
time: 1.40 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8786224, 0.8474249, -0.3850430, 0.4232965, -1.3019189, 1.2324679
1: -0.6006573, 0.6250733, -0.2390265, 0.2501809, -0.8508382, 0.8640998
2: -0.6259792, 0.8989227, -0.2391254, 0.3738037, -0.9997829, 1.1380482
3: -0.6627718, 0.6561813, -0.2280319, 0.3784219, -1.0411937, 0.8842131
4: -0.7987170, 0.7769273, -0.3544517, 0.3122852, -1.1110022, 1.1313790
5: -0.3523200, 1.1818788, 0.3691384, 1.0895119, -1.4418318, 0.8127404
6: -0.6452506, 0.7721056, -0.2926069, 0.3896344, -1.0348849, 1.0647124
7: -0.6896441, 0.7994627, -0.3184187, 0.2778150, -0.9674591, 1.1178813
8: -0.8720770, 0.8647815, -0.3808998, 0.4717198, -1.3437967, 1.2456813
9: -0.7284738, 0.7906576, -0.2773459, 0.3712761, -1.0997499, 1.0680034

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7814618, upper bound: 1.7511661
time: 1.46 seconds

## Relational analysis of NS_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7678959, upper bound: 1.7511661
time: 1.43 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2.0402651, 1.5094993, -0.3602495, 0.3956605, -2.4359255, 1.8697488
1: -1.3782039, 1.4663388, -0.2222115, 0.2329190, -1.6111228, 1.6885502
2: -1.6839046, 1.6868033, -0.2218424, 0.3455530, -2.0294576, 1.9086457
3: -2.1017399, 1.0904381, -0.2125327, 0.3538914, -2.4556313, 1.3029708
4: -1.9013803, 1.6914231, -0.3283492, 0.2925924, -2.1939726, 2.0197723
5: -1.7476531, 1.5361416, 0.4065565, 1.0835268, -2.8311801, 1.1295851
6: -1.4674358, 1.6727424, -0.2734039, 0.3647295, -1.8321654, 1.9461462
7: -1.5767463, 1.7379744, -0.2996936, 0.2588067, -1.8355531, 2.0376680
8: -2.0509739, 1.4923555, -0.3521806, 0.4455785, -2.4965525, 1.8445361
9: -1.6773412, 1.5785158, -0.2567365, 0.3472790, -2.0246203, 1.8352523

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7861047, upper bound: 1.7903640
time: 1.36 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
time: 1.36 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3209792, 0.3489072, -0.4347813, 0.4715886, -0.7925678, 0.7836884
1: -0.1935983, 0.2055060, -0.2770618, 0.2873988, -0.4809970, 0.4825678
2: -0.1932847, 0.3014991, -0.2783036, 0.4242058, -0.6174904, 0.5798028
3: -0.1860236, 0.3131845, -0.2559901, 0.4203966, -0.6064203, 0.5691746
4: -0.2907199, 0.2589112, -0.4049861, 0.3674742, -0.6581942, 0.6638972
5: 0.4704674, 1.0772523, 0.2901687, 1.1002619, -0.6297945, 0.7870836
6: -0.2426971, 0.3227254, -0.3282078, 0.4404147, -0.6831118, 0.6509331
7: -0.2686645, 0.2308226, -0.3526970, 0.3383311, -0.6069956, 0.5835196
8: -0.3069575, 0.4025055, -0.4308633, 0.5264704, -0.8334279, 0.8333688
9: -0.2262427, 0.3062355, -0.3359410, 0.4120143, -0.6382570, 0.6421765

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8013404
time: 1.54 seconds

## Relational analysis of NS_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7951458
time: 1.39 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3447450, 0.3770972, -0.5134679, 0.5645566, -0.9093016, 0.8905652
1: -0.2109341, 0.2220588, -0.3459426, 0.3644861, -0.5754203, 0.5680014
2: -0.2103595, 0.3280855, -0.3461093, 0.5270848, -0.7374443, 0.6741948
3: -0.2020898, 0.3377175, -0.3100919, 0.4820551, -0.6841449, 0.6478094
4: -0.3130799, 0.2793242, -0.4823438, 0.4645006, -0.7775805, 0.7616680
5: 0.4317333, 1.0805568, 0.1420560, 1.1159706, -0.6842374, 0.9385008
6: -0.2610384, 0.3481826, -0.3921979, 0.5173920, -0.7784303, 0.7403805
7: -0.2871882, 0.2476684, -0.4173534, 0.4478665, -0.7350547, 0.6650218
8: -0.3338551, 0.4286104, -0.5213739, 0.6099824, -0.9438375, 0.9499843
9: -0.2445538, 0.3311106, -0.4323417, 0.4945228, -0.7390767, 0.7634523

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8018666, upper bound: 1.7951458
time: 1.68 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7951458
time: 1.54 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.3476999, 0.3806020, -0.5643247, 0.6300236, -0.9777235, 0.9449267
1: -0.2130895, 0.2241170, -0.3961554, 0.4120139, -0.6251035, 0.6202724
2: -0.2124825, 0.3313911, -0.3854850, 0.6126971, -0.8251796, 0.7168761
3: -0.2040873, 0.3407677, -0.3498739, 0.5193863, -0.7234737, 0.6906416
4: -0.3158599, 0.2818623, -0.5349531, 0.5200950, -0.8359548, 0.8168155
5: 0.4269174, 1.0809677, 0.0421045, 1.1253008, -0.6983834, 1.0388632
6: -0.2633188, 0.3513476, -0.4347078, 0.5638385, -0.8271574, 0.7860553
7: -0.2894911, 0.2497627, -0.4617690, 0.5192186, -0.8087096, 0.7115318
8: -0.3371992, 0.4318563, -0.5812346, 0.6646393, -1.0018384, 1.0130908
9: -0.2468306, 0.3342032, -0.4888602, 0.5543125, -0.8011431, 0.8230634

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A2_A1_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8018571, upper bound: 1.7951458
time: 1.69 seconds

## Relational analysis of NS_B2_A1_B1_A2_A1_A2

### Relational analysis result of NS_B2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7951458
time: 1.42 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.3963786, 0.4343500, -0.6020528, 0.6734658, -1.0698445, 1.0364027
1: -0.2468828, 0.2582774, -0.4300621, 0.4450061, -0.6918889, 0.6883396
2: -0.2468674, 0.3856780, -0.4118789, 0.6713426, -0.9182099, 0.7975569
3: -0.2344479, 0.3883056, -0.3762205, 0.5459449, -0.7803928, 0.7645261
4: -0.3663573, 0.3221762, -0.5733695, 0.5569133, -0.9232706, 0.8955457
5: 0.3523340, 1.0920444, -0.0248424, 1.1317575, -0.7794235, 1.1168869
6: -0.3007571, 0.4007336, -0.4634137, 0.5970453, -0.8978024, 0.8641472
7: -0.3259081, 0.2903583, -0.4912835, 0.5677858, -0.8936939, 0.7816418
8: -0.3923866, 0.4835427, -0.6208780, 0.7047341, -1.0971206, 1.1044207
9: -0.2891463, 0.3808738, -0.5272510, 0.5943489, -0.8834952, 0.9081248

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A2_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8018563, upper bound: 1.7951458
time: 1.59 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7951458
time: 1.33 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3268338, 0.3558514, -0.6858145, 0.7418243, -1.0686581, 1.0416659
1: -0.1978688, 0.2095836, -0.4885908, 0.5058686, -0.7037374, 0.6981744
2: -0.1974908, 0.3080485, -0.4735324, 0.7644120, -0.9619029, 0.7815809
3: -0.1899813, 0.3192281, -0.4547266, 0.5880598, -0.7780410, 0.7739547
4: -0.2962283, 0.2639399, -0.6425705, 0.6285430, -0.9247712, 0.9065104
5: 0.4609255, 1.0780662, -0.1344290, 1.1471356, -0.6862102, 1.2124953
6: -0.2472154, 0.3289964, -0.5153694, 0.6558409, -0.9030563, 0.8443657
7: -0.2732277, 0.2349724, -0.5570984, 0.6489267, -0.9221544, 0.7920709
8: -0.3135834, 0.4089362, -0.6957597, 0.7688834, -1.0824668, 1.1046959
9: -0.2307535, 0.3123634, -0.5957745, 0.6639369, -0.8946904, 0.9081379

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8013227
time: 1.39 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.37 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3505303, 0.3840135, -0.9265468, 0.8718569, -1.2223872, 1.3105602
1: -0.2151541, 0.2261109, -0.6291366, 0.6570483, -0.8722024, 0.8552475
2: -0.2146026, 0.3345574, -0.6658198, 0.9307771, -1.1453798, 1.0003772
3: -0.2060008, 0.3437297, -0.7178342, 0.6733859, -0.8793867, 1.0615638
4: -0.3186458, 0.2842933, -0.8401027, 0.8125520, -1.1311978, 1.1243960
5: 0.4223042, 1.0815150, -0.4073130, 1.1941018, -0.7717976, 1.4888279
6: -0.2656020, 0.3543794, -0.6779570, 0.8034403, -1.0690423, 1.0323364
7: -0.2918019, 0.2517692, -0.7223284, 0.8363782, -1.1281800, 0.9740976
8: -0.3405572, 0.4349654, -0.9179788, 0.8885338, -1.2290909, 1.3529443
9: -0.2490527, 0.3371661, -0.7637035, 0.8224760, -1.0715287, 1.1008695

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8013105
time: 1.57 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.32 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.3554177, 0.3899045, -1.1090240, 0.9721782, -1.3275958, 1.4989285
1: -0.2187191, 0.2295544, -0.7486025, 0.7853636, -1.0040827, 0.9781569
2: -0.2182644, 0.3400251, -0.8283271, 1.0545540, -1.2728183, 1.1683521
3: -0.2093046, 0.3488445, -0.9410976, 0.7400413, -0.9493459, 1.2899421
4: -0.3234577, 0.2884911, -1.0109348, 0.9528694, -1.2763271, 1.2994260
5: 0.4143391, 1.0824621, -0.6248754, 1.2475197, -0.8331807, 1.7073375
6: -0.2695458, 0.3596144, -0.8054366, 0.9398800, -1.2094257, 1.1650510
7: -0.2957935, 0.2552333, -0.8566403, 0.9813437, -1.2771373, 1.1118736
8: -0.3463573, 0.4403334, -1.1018828, 0.9854598, -1.3318172, 1.5422162
9: -0.2528902, 0.3422810, -0.9106783, 0.9448593, -1.1977495, 1.2529593

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8012567
time: 1.62 seconds

## Relational analysis of NS_B2_A1_B2_A2_A1_B2

### Relational analysis result of NS_B2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.36 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.4061598, 0.4435579, -1.2305369, 1.0419328, -1.4480926, 1.6740949
1: -0.2544653, 0.2650220, -0.8307322, 0.8737678, -1.1282331, 1.0957541
2: -0.2545951, 0.3955703, -0.9398335, 1.1366322, -1.3912272, 1.3354039
3: -0.2399585, 0.3965395, -1.0922076, 0.7855300, -1.0254885, 1.4887470
4: -0.3762755, 0.3331810, -1.1281443, 1.0487487, -1.4250243, 1.4613253
5: 0.3367803, 1.0941542, -0.7716962, 1.2849884, -0.9482081, 1.8658504
6: -0.3075467, 0.4107426, -0.8935922, 1.0350201, -1.3425667, 1.3043348
7: -0.3323733, 0.3026759, -0.9501243, 1.0795853, -1.4119586, 1.2528002
8: -0.4019556, 0.4943068, -1.2258166, 1.0512642, -1.4532198, 1.7201234
9: -0.3005819, 0.3888692, -1.0102066, 1.0273356, -1.3279176, 1.3990757

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8013228
time: 1.54 seconds

## Relational analysis of NS_B2_A1_B2_A2_A2_B2

### Relational analysis result of NS_B2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.35 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5149411, 0.5665838, -0.7150925, 0.7606291, -1.2755702, 1.2816763
1: -0.3473791, 0.3659265, -0.5056040, 0.5246340, -0.8720132, 0.8715305
2: -0.3473086, 0.5297052, -0.4956166, 0.7880920, -1.1354005, 1.0253217
3: -0.3113397, 0.4831579, -0.4847069, 0.5997485, -0.9110882, 0.9678649
4: -0.4838515, 0.4662447, -0.6658015, 0.6514546, -1.1353061, 1.1320462
5: 0.1390567, 1.1162539, -0.1670958, 1.1524032, -1.0133466, 1.2833496
6: -0.3934655, 0.5187674, -0.5339166, 0.6736562, -1.0671217, 1.0526839
7: -0.4187163, 0.4500682, -0.5789045, 0.6736021, -1.0923184, 1.0289726
8: -0.5232513, 0.6114459, -0.7208554, 0.7857795, -1.3090308, 1.3323013
9: -0.4340828, 0.4962885, -0.6172895, 0.6844630, -1.1185459, 1.1135781

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_B2_A2_A1_B1_A1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.28 seconds

## Relational analysis of NS_B2_A2_A1_B1_A1_A2

### Relational analysis result of NS_B2_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.24 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.6439842, 0.7144624, -0.7310996, 0.7700778, -1.4140620, 1.4455620
1: -0.4637590, 0.4784979, -0.5148451, 0.5347475, -0.9985065, 0.9933430
2: -0.4427387, 0.7273628, -0.5080263, 0.7999779, -1.2427166, 1.2353890
3: -0.4113853, 0.5710137, -0.5020613, 0.6056521, -1.0170374, 1.0730751
4: -0.6115010, 0.5952169, -0.6785498, 0.6640313, -1.2755322, 1.2737668
5: -0.0884317, 1.1394757, -0.1854379, 1.1550397, -1.2434714, 1.3249135
6: -0.4918039, 0.6300656, -0.5455297, 0.6830835, -1.1748874, 1.1755953
7: -0.5247046, 0.6142172, -0.5897163, 0.6872296, -1.2119342, 1.2039335
8: -0.6612973, 0.7427015, -0.7353725, 0.7941901, -1.4554875, 1.4780741
9: -0.5644790, 0.6345381, -0.6287247, 0.6955552, -1.2600341, 1.2632627

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B2_A2_A1_B1_A2_A1

### Relational analysis result of NS_B2_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7997147, upper bound: 1.8015216
time: 1.49 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2_A2

### Relational analysis result of NS_B2_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015216, upper bound: 1.8015216
time: 1.41 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.5149411, 0.5665838, -1.4563478, 1.1585714, -1.6735125, 2.0229316
1: -0.3473791, 0.3659265, -0.9808577, 1.0228373, -1.3702164, 1.3467842
2: -0.3473086, 0.5297052, -1.1369269, 1.2894864, -1.6367950, 1.6666322
3: -0.3113397, 0.4831579, -1.3654125, 0.8637218, -1.1750615, 1.8485703
4: -0.4838515, 0.4662447, -1.3331177, 1.2144201, -1.6982715, 1.7993624
5: 0.1390567, 1.1162539, -1.0372155, 1.3437510, -1.2046943, 2.1534693
6: -0.3934655, 0.5187674, -1.0431294, 1.2008965, -1.5943620, 1.5618968
7: -0.4187163, 0.4500682, -1.1088476, 1.2509400, -1.6696563, 1.5589159
8: -0.5232513, 0.6114459, -1.4489088, 1.1704319, -1.6936831, 2.0603547
9: -0.4340828, 0.4962885, -1.1883422, 1.1763136, -1.6103965, 1.6846306

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_B2_A2_A1_B2_A1_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.29 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_A2

### Relational analysis result of NS_B2_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.40 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.6439842, 0.7144624, -1.4777822, 1.1702266, -1.8142108, 2.1922445
1: -0.4637590, 0.4784979, -0.9952224, 1.0376430, -1.5014019, 1.4737203
2: -0.4427387, 0.7273628, -1.1560742, 1.3039964, -1.7467351, 1.8834369
3: -0.4113853, 0.5710137, -1.3916755, 0.8714217, -1.2828070, 1.9626892
4: -0.6115010, 0.5952169, -1.3531065, 1.2307181, -1.8422191, 1.9483234
5: -0.0884317, 1.1394757, -1.0627352, 1.3498073, -1.4382390, 2.2022109
6: -0.4918039, 0.6300656, -1.0579075, 1.2171001, -1.7089040, 1.6879730
7: -0.5247046, 0.6142172, -1.1245697, 1.2677228, -1.7924274, 1.7387868
8: -0.6612973, 0.7427015, -1.4704027, 1.1818736, -1.8431709, 2.2131042
9: -0.5644790, 0.6345381, -1.2055591, 1.1906333, -1.7551123, 1.8400972

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015215, upper bound: 1.7995265
time: 1.76 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015216, upper bound: 1.8014497
time: 1.49 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.4599214, 1.1736127, -0.5149411, 0.5665838, -2.0265052, 1.6885538
1: -0.9857737, 1.0406520, -0.3473791, 0.3659265, -1.3517002, 1.3880312
2: -1.1503291, 1.2915759, -0.3473086, 0.5297052, -1.6800344, 1.6388845
3: -1.3774662, 0.8714005, -0.3113397, 0.4831579, -1.8606241, 1.1827402
4: -1.3494058, 1.2297438, -0.4838515, 0.4662447, -1.8156505, 1.7135953
5: -1.0488572, 1.3557205, 0.1390567, 1.1162539, -2.1651111, 1.2166638
6: -1.0600083, 1.2146206, -0.3934655, 0.5187674, -1.5787756, 1.6080861
7: -1.1265986, 1.2650421, -0.4187163, 0.4500682, -1.5766668, 1.6837584
8: -1.4597722, 1.1754880, -0.5232513, 0.6114459, -2.0712180, 1.6987393
9: -1.1980908, 1.1830301, -0.4340828, 0.4962885, -1.6943793, 1.6171130

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of NS_B2_A2_A2_B1_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.22 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.36 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1.4815611, 1.1860344, -0.6439842, 0.7144624, -2.1960235, 1.8300186
1: -1.0003991, 1.0563956, -0.4637590, 0.4784979, -1.4788970, 1.5201547
2: -1.1701858, 1.3061929, -0.4427387, 0.7273628, -1.8975486, 1.7489315
3: -1.4043756, 0.8795012, -0.4113853, 0.5710137, -1.9753892, 1.2908865
4: -1.3702786, 1.2468179, -0.6115010, 0.5952169, -1.9654955, 1.8583189
5: -1.0750031, 1.3623930, -0.0884317, 1.1394757, -2.2144790, 1.4508247
6: -1.0757072, 1.2315633, -0.4918039, 0.6300656, -1.7057728, 1.7233672
7: -1.1432462, 1.2825372, -0.5247046, 0.6142172, -1.7574633, 1.8072418
8: -1.4818425, 1.1872070, -0.6612973, 0.7427015, -2.2245440, 1.8485043
9: -1.2158148, 1.1977177, -0.5644790, 0.6345381, -1.8503529, 1.7621967

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B2_A2_A2_B1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7995265, upper bound: 1.8015215
time: 1.61 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8014497, upper bound: 1.8015216
time: 1.32 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.9619480, 0.8901958, -1.4563478, 1.1585714, -2.1205194, 2.3465438
1: -0.6506742, 0.6813376, -0.9808577, 1.0228373, -1.6735115, 1.6621953
2: -0.6959093, 0.9547869, -1.1369269, 1.2894864, -1.9853957, 2.0917139
3: -0.7600281, 0.6861064, -1.3654125, 0.8637218, -1.6237500, 2.0515189
4: -0.8721974, 0.8388641, -1.3331177, 1.2144201, -2.0866175, 2.1719818
5: -0.4487125, 1.2040765, -1.0372155, 1.3437510, -1.7924634, 2.2412920
6: -0.7022405, 0.8281615, -1.0431294, 1.2008965, -1.9031370, 1.8712909
7: -0.7468738, 0.8638758, -1.1088476, 1.2509400, -1.9978137, 1.9727235
8: -0.9531361, 0.9068393, -1.4489088, 1.1704319, -2.1235681, 2.3557482
9: -0.7914150, 0.8460345, -1.1883422, 1.1763136, -1.9677286, 2.0343766

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 165

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_B2_A2_A2_B2_A1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7175385, upper bound: 1.6696560
time: 1.26 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6697041, upper bound: 1.6696560
time: 1.02 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.3451045, 1.1077013, -1.4777822, 1.1702266, -2.5153310, 2.5854836
1: -0.9081688, 0.9571195, -0.9952224, 1.0376430, -1.9458117, 1.9523420
2: -1.0449669, 1.2140200, -1.1560742, 1.3039964, -2.3489633, 2.3700943
3: -1.2346816, 0.8284187, -1.3916755, 0.8714217, -2.1061034, 2.2200942
4: -1.2386547, 1.1391478, -1.3531065, 1.2307181, -2.4693727, 2.4922543
5: -0.9101260, 1.3203161, -1.0627352, 1.3498073, -2.2599332, 2.3830514
6: -0.9767103, 1.1247227, -1.0579075, 1.2171001, -2.1938105, 2.1826301
7: -1.0382655, 1.1722132, -1.1245697, 1.2677228, -2.3059883, 2.2967830
8: -1.3426676, 1.1133087, -1.4704027, 1.1818736, -2.5245411, 2.5837114
9: -1.1040466, 1.1050985, -1.2055591, 1.1906333, -2.2946799, 2.3106575

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_B2_A2_A2_B2_A2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7174693, upper bound: 1.6696560
time: 1.63 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6696560, upper bound: 1.6696560
time: 1.19 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.87 seconds
NS_B1_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7814949, upper bound: 1.7511661
NS_B1_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7679741, upper bound: 1.7511661
NS_B1_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7740677, upper bound: 1.7511661
NS_B1_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7511661
NS_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7843712
NS_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7703235
NS_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7763739
NS_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7511661
NS_B1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7843712
NS_B1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7702354
NS_B1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7763739
NS_B1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7511661
NS_B1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7843712
NS_B1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7702354
NS_B1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7763739
NS_B1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7511661
NS_B1_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7813746, upper bound: 1.7511661
NS_B1_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7678785, upper bound: 1.7511661
NS_B1_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7738816, upper bound: 1.7511661
NS_B1_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7511661
NS_B1_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7813746, upper bound: 1.7511661
NS_B1_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7678785, upper bound: 1.7511661
NS_B1_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7861047, upper bound: 1.7903640
NS_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
NS_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7812601, upper bound: 1.7511661
NS_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7675506, upper bound: 1.7511661
NS_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7861047, upper bound: 1.7903640
NS_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
NS_B1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7814618, upper bound: 1.7511661
NS_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7678959, upper bound: 1.7511661
NS_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7861047, upper bound: 1.7903640
NS_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
NS_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8013404
NS_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7951458
NS_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8018666, upper bound: 1.7951458
NS_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7951458
NS_B2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8018571, upper bound: 1.7951458
NS_B2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7951458
NS_B2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8018563, upper bound: 1.7951458
NS_B2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7951458
NS_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8013227
NS_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8013105
NS_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_B2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8012567
NS_B2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_B2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.8013228
NS_B2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_B2_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7997147, upper bound: 1.8015216
NS_B2_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8015216, upper bound: 1.8015216
NS_B2_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8015215, upper bound: 1.7995265
NS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8015216, upper bound: 1.8014497
NS_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7995265, upper bound: 1.8015215
NS_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.8014497, upper bound: 1.8015216
NS_B2_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7175385, upper bound: 1.6696560
NS_B2_A2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.6697041, upper bound: 1.6696560
NS_B2_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.7174693, upper bound: 1.6696560
NS_B2_A2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 5, lower bound: -1.6696560, upper bound: 1.6696560

## BFS NS instance: NS_B1_A1_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.3396049, 0.3710002, -0.3068123, 0.3321023, -0.6717072, 0.6778125
1: -0.2071847, 0.2184786, -0.1832643, 0.1956385, -0.4028232, 0.4017429
2: -0.2066664, 0.3223355, -0.1831059, 0.2856508, -0.4923172, 0.5054414
3: -0.1986149, 0.3324115, -0.1764463, 0.2985602, -0.4971751, 0.5088578
4: -0.3082437, 0.2749094, -0.2773910, 0.2467429, -0.5549866, 0.5523004
5: 0.4401106, 1.0798421, 0.4935573, 1.0752822, -0.6351717, 0.5862848
6: -0.2570716, 0.3426765, -0.2317638, 0.3075498, -0.5646213, 0.5744402
7: -0.2831817, 0.2440248, -0.2576224, 0.2207806, -0.5039623, 0.5016472
8: -0.3280376, 0.4229645, -0.2909233, 0.3869441, -0.7149817, 0.7138878
9: -0.2405936, 0.3257304, -0.2153271, 0.2914075, -0.5320010, 0.5410575

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A1_A1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7663792, upper bound: 1.7355502
time: 1.60 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7730046, upper bound: 1.7416129
time: 1.51 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.6250685, 0.7073832, -0.2741010, 0.2924515, -0.9175200, 0.9814842
1: -0.4118747, 0.4151300, -0.1588809, 0.1723602, -0.5842349, 0.5740109
2: -0.4156099, 0.6551794, -0.1590893, 0.2492907, -0.6649006, 0.8142688
3: -0.3873562, 0.6299215, -0.1541860, 0.2640539, -0.6514101, 0.7841074
4: -0.6036654, 0.5147159, -0.2460925, 0.2180309, -0.8216963, 0.7608083
5: -0.0149268, 1.1488509, 0.5480386, 1.0708894, -1.0858161, 0.6008123
6: -0.4826300, 0.6417401, -0.2066396, 0.2717433, -0.7543733, 0.8483797
7: -0.5109049, 0.4627107, -0.2315684, 0.1981321, -0.7090371, 0.6942791
8: -0.6761220, 0.7296386, -0.2532819, 0.3502261, -1.0263481, 0.9829205
9: -0.4703666, 0.6179531, -0.1895713, 0.2566932, -0.7270598, 0.8075244

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A1_A1_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7416853, upper bound: 1.7139773
time: 1.58 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7355682, upper bound: 1.7139773
time: 1.59 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.5735063, 0.6167204, -0.2852308, 0.3060575, -0.8795637, 0.9019513
1: -0.3865846, 0.4150793, -0.1672477, 0.1803475, -0.5669321, 0.5823270
2: -0.3935111, 0.5629971, -0.1673304, 0.2616305, -0.6551416, 0.7303275
3: -0.3336925, 0.5361792, -0.1617797, 0.2758944, -0.6095869, 0.6979589
4: -0.5441424, 0.5340623, -0.2568123, 0.2278831, -0.7720255, 0.7908745
5: 0.0598775, 1.1298647, 0.5293438, 1.0723630, -1.0124855, 0.6005208
6: -0.4351360, 0.5842324, -0.2151716, 0.2840299, -0.7191659, 0.7994040
7: -0.4600056, 0.5111479, -0.2405086, 0.2057653, -0.6657709, 0.7516565
8: -0.5791163, 0.6823637, -0.2661728, 0.3628254, -0.9419417, 0.9485366
9: -0.5162724, 0.5241954, -0.1984092, 0.2685689, -0.7848413, 0.7226046

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A1_A2_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7475049, upper bound: 1.7139773
time: 1.52 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7397975, upper bound: 1.7139773
time: 1.53 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.9399180, 1.0000548, -0.2525264, 0.2665746, -1.2064927, 1.2525812
1: -0.6758645, 0.7523189, -0.1429237, 0.1568770, -0.8327415, 0.8952425
2: -0.6978070, 0.9295834, -0.1432734, 0.2253714, -0.9231784, 1.0728568
3: -0.5389270, 0.8419935, -0.1394659, 0.2417149, -0.7806419, 0.9814593
4: -0.9116932, 0.9740680, -0.2256273, 0.1989327, -1.1106260, 1.1996953
5: -0.5483860, 1.2080537, 0.5840236, 1.0680329, -1.6164188, 0.6240301
6: -0.7175627, 0.9640954, -0.1902675, 0.2479258, -0.9654886, 1.1543629
7: -0.7434371, 0.9676052, -0.2144606, 0.1833354, -0.9267724, 1.1820657
8: -0.9706945, 1.0941219, -0.2284323, 0.3258026, -1.2964971, 1.3225542
9: -0.9925785, 0.8204971, -0.1724396, 0.2338717, -1.2264502, 0.9929367

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A1_A2_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7322638, upper bound: 1.7139773
time: 1.47 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139215, upper bound: 1.7139773
time: 1.22 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.4858801, 0.5260668, -0.2621492, 0.2778409, -0.7637210, 0.7882161
1: -0.3187082, 0.3371679, -0.1498958, 0.1637830, -0.4824911, 0.4870638
2: -0.3233421, 0.4774862, -0.1502395, 0.2360399, -0.5593821, 0.6277257
3: -0.2865390, 0.4613173, -0.1460315, 0.2513385, -0.5378776, 0.6073488
4: -0.4542625, 0.4313959, -0.2345812, 0.2074510, -0.6617135, 0.6659771
5: 0.1989636, 1.1107183, 0.5681140, 1.0693069, -0.8703433, 0.5426042
6: -0.3682292, 0.4912775, -0.1974777, 0.2585490, -0.6267781, 0.6887552
7: -0.3914780, 0.4065697, -0.2219679, 0.1899350, -0.5814130, 0.6285376
8: -0.4858544, 0.5821977, -0.2394387, 0.3366961, -0.8225505, 0.8216364
9: -0.3996980, 0.4610791, -0.1800807, 0.2439405, -0.6436386, 0.6411598

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7365488, upper bound: 1.7704756
time: 1.38 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7417652, upper bound: 1.7763012
time: 1.33 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.4297549, 0.4663302, -0.5296632, 0.6048687, -1.0346236, 0.9959934
1: -0.2730935, 0.2827727, -0.3510036, 0.3557634, -0.6288569, 0.6337763
2: -0.2741294, 0.4191768, -0.3483216, 0.5326291, -0.8067585, 0.7674984
3: -0.2531745, 0.4162017, -0.3285526, 0.5359382, -0.7891127, 0.7447543
4: -0.3999440, 0.3614382, -0.4922375, 0.4442576, -0.8442016, 0.8536756
5: 0.2985128, 1.0991895, 0.1187691, 1.1047258, -0.8062131, 0.9804204
6: -0.3243336, 0.4352039, -0.4025492, 0.5538714, -0.8782049, 0.8377531
7: -0.3488091, 0.3320694, -0.4368541, 0.3734057, -0.7222148, 0.7689234
8: -0.4254918, 0.5208219, -0.5492854, 0.6395336, -1.0650253, 1.0701073
9: -0.3294070, 0.4079497, -0.3925054, 0.5293832, -0.8587902, 0.8004552

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A2_B1_B2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7455782
time: 1.49 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_B2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7387646
time: 1.32 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.4521858, 0.4892803, -0.4499175, 0.5018501, -0.9540359, 0.9391978
1: -0.2904126, 0.3040384, -0.2876516, 0.2953118, -0.5857245, 0.5916899
2: -0.2933790, 0.4411240, -0.2859229, 0.4457414, -0.7391204, 0.7270469
3: -0.2658479, 0.4345105, -0.2731892, 0.4462850, -0.7121329, 0.7076997
4: -0.4219492, 0.3889593, -0.4120311, 0.3696598, -0.7916089, 0.8009904
5: 0.2603597, 1.1038705, 0.2603196, 1.0951811, -0.8348214, 0.8435509
6: -0.3415779, 0.4579461, -0.3422053, 0.4608407, -0.8024186, 0.8001514
7: -0.3657780, 0.3607229, -0.3691618, 0.3222173, -0.6879953, 0.7298846
8: -0.4489353, 0.5456309, -0.4528883, 0.5441353, -0.9930706, 0.9985192
9: -0.3579234, 0.4273470, -0.3255886, 0.4411913, -0.7991146, 0.7529356

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7509678
time: 1.27 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7425660
time: 1.64 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.3975498, 0.4354789, -0.7314250, 0.8357669, -1.2333167, 1.1669040
1: -0.2477533, 0.2591043, -0.4929958, 0.4913825, -0.7391358, 0.7521001
2: -0.2476964, 0.3868912, -0.4881776, 0.7606623, -1.0083587, 0.8750688
3: -0.2351163, 0.3893153, -0.4634953, 0.7368799, -0.9719962, 0.8528105
4: -0.3675734, 0.3233073, -0.6768867, 0.6114547, -0.9790282, 1.0001941
5: 0.3505129, 1.0923032, -0.1984910, 1.1343246, -0.7838117, 1.2907941
6: -0.3015895, 0.4018967, -0.5594593, 0.7623839, -1.0639734, 0.9613560
7: -0.3266729, 0.2918688, -0.5885748, 0.5217570, -0.8484299, 0.8804436
8: -0.3935598, 0.4847705, -0.7714952, 0.8533524, -1.2469122, 1.2562656
9: -0.2904637, 0.3818540, -0.5424880, 0.7358369, -1.0263005, 0.9243420

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7338122
time: 2.54 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139773
time: 1.34 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.5286555, 0.5856364, -0.2684951, 0.2855985, -0.8142540, 0.8541315
1: -0.3616810, 0.3793340, -0.1546664, 0.1683368, -0.5300179, 0.5340005
2: -0.3586897, 0.5540992, -0.1549382, 0.2430756, -0.6017653, 0.7090374
3: -0.3229548, 0.4934987, -0.1503612, 0.2580896, -0.5810444, 0.6438599
4: -0.4980999, 0.4824762, -0.2406930, 0.2130684, -0.7111683, 0.7231692
5: 0.1102437, 1.1188924, 0.5574552, 1.0701472, -0.9599035, 0.5614372
6: -0.4056692, 0.5315722, -0.2023421, 0.2655547, -0.6712239, 0.7339143
7: -0.4316125, 0.4705589, -0.2270651, 0.1942873, -0.6258998, 0.6976241
8: -0.5407287, 0.6261013, -0.2467887, 0.3438797, -0.8846084, 0.8728900
9: -0.4502872, 0.5134728, -0.1851199, 0.2507114, -0.7009985, 0.6985927

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B1_B1_B1_A1

### Relational analysis result of NS_B1_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7365488, upper bound: 1.7703020
time: 1.47 seconds

## Relational analysis of NS_B1_A1_B2_B1_B1_B1_A2

### Relational analysis result of NS_B1_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7417652, upper bound: 1.7763001
time: 1.35 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.4677511, 0.5047973, -0.5170811, 0.5894875, -1.0572386, 1.0218785
1: -0.3021225, 0.3192862, -0.3415449, 0.3467340, -0.6488565, 0.6608311
2: -0.3072280, 0.4559634, -0.3390053, 0.5186795, -0.8259075, 0.7949687
3: -0.2747287, 0.4468897, -0.3199680, 0.5225527, -0.7972814, 0.7668577
4: -0.4368275, 0.4085187, -0.4801192, 0.4331197, -0.8699471, 0.8886379
5: 0.2331600, 1.1070354, 0.1399035, 1.1030599, -0.8698999, 0.9671320
6: -0.3535084, 0.4733228, -0.3929040, 0.5399814, -0.8934897, 0.8662268
7: -0.3772512, 0.3811672, -0.4267474, 0.3647763, -0.7420275, 0.8079146
8: -0.4647862, 0.5625319, -0.5347124, 0.6252904, -1.0900767, 1.0972444
9: -0.3772040, 0.4418023, -0.3825144, 0.5159578, -0.8931618, 0.8243166

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B2_B1_B1_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7455782
time: 1.30 seconds

## Relational analysis of NS_B1_A1_B2_B1_B1_B2_B2

### Relational analysis result of NS_B1_A1_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7387646
time: 1.33 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.4917094, 0.5342917, -0.4364847, 0.4859165, -0.9776258, 0.9707764
1: -0.3245249, 0.3429850, -0.2778531, 0.2859556, -0.6104805, 0.6208382
2: -0.3282074, 0.4880323, -0.2762716, 0.4307141, -0.7589215, 0.7643039
3: -0.2914653, 0.4657325, -0.2641081, 0.4324185, -0.7238837, 0.7298406
4: -0.4601792, 0.4384705, -0.3993928, 0.3581221, -0.8183013, 0.8378634
5: 0.1868260, 1.1118304, 0.2822129, 1.0933133, -0.9064873, 0.8296174
6: -0.3733162, 0.4968581, -0.3318386, 0.4464515, -0.8197678, 0.8286967
7: -0.3970074, 0.4153199, -0.3586919, 0.3126957, -0.7097031, 0.7740117
8: -0.4934260, 0.5881350, -0.4376850, 0.5293803, -1.0228063, 1.0258200
9: -0.4066411, 0.4681631, -0.3152386, 0.4271317, -0.8337727, 0.7834017

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B2_B1_B2_B1_B1

### Relational analysis result of NS_B1_A1_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7509678
time: 1.46 seconds

## Relational analysis of NS_B1_A1_B2_B1_B2_B1_B2

### Relational analysis result of NS_B1_A1_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7425548
time: 1.41 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.4339488, 0.4707178, -0.6994669, 0.7978587, -1.2318076, 1.1701847
1: -0.2764047, 0.2866328, -0.4696839, 0.4691236, -0.7455283, 0.7563167
2: -0.2776124, 0.4233729, -0.4652165, 0.7249107, -1.0025232, 0.8885894
3: -0.2555238, 0.4197018, -0.4418908, 0.7038900, -0.9594138, 0.8615926
4: -0.4041510, 0.3664750, -0.6468188, 0.5840050, -0.9881560, 1.0132937
5: 0.2915503, 1.1000844, -0.1464044, 1.1298808, -0.8383304, 1.2464888
6: -0.3275662, 0.4395519, -0.5347953, 0.7281513, -1.0557175, 0.9743472
7: -0.3520533, 0.3372944, -0.5636658, 0.4991041, -0.8511574, 0.9009602
8: -0.4299739, 0.5255349, -0.7353252, 0.8182484, -1.2482224, 1.2608601
9: -0.3348591, 0.4113412, -0.5178643, 0.7023871, -1.0372462, 0.9292055

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B2_B1_B2_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7338122
time: 1.29 seconds

## Relational analysis of NS_B1_A1_B2_B1_B2_B2_B2

### Relational analysis result of NS_B1_A1_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7135653
time: 1.23 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.5628905, 0.6283717, -0.3137490, 0.3403305, -0.9032210, 0.9421207
1: -0.3948663, 0.4107593, -0.1883242, 0.2004700, -0.5953364, 0.5990835
2: -0.3844815, 0.6104673, -0.1880898, 0.2934107, -0.6778923, 0.7985570
3: -0.3488725, 0.5183766, -0.1811357, 0.3057208, -0.6545933, 0.6995123
4: -0.5334927, 0.5186954, -0.2839174, 0.2527009, -0.7861936, 0.8026127
5: 0.0446496, 1.1250556, 0.4822516, 1.0762466, -1.0315970, 0.6428039
6: -0.4336163, 0.5625763, -0.2371172, 0.3149801, -0.7485964, 0.7996935
7: -0.4606466, 0.5173721, -0.2630290, 0.2256976, -0.6863441, 0.7804012
8: -0.5797271, 0.6631154, -0.2987741, 0.3945637, -0.9742907, 0.9618894
9: -0.4874005, 0.5527905, -0.2206717, 0.2986680, -0.7860684, 0.7734621

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B2_B1_B1_A1

### Relational analysis result of NS_B1_A1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7365488, upper bound: 1.7703020
time: 1.38 seconds

## Relational analysis of NS_B1_A1_B2_B2_B1_B1_A2

### Relational analysis result of NS_B1_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7417652, upper bound: 1.7763001
time: 1.25 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.4962688, 0.5406715, -0.5857134, 0.6629274, -1.1591963, 1.1263849
1: -0.3290362, 0.3475174, -0.3867072, 0.3898940, -0.7189302, 0.7342246
2: -0.3319809, 0.4962563, -0.3834882, 0.5976552, -0.9296362, 0.8797445
3: -0.2953915, 0.4691567, -0.3649905, 0.5864645, -0.8818560, 0.8341472
4: -0.4648118, 0.4439573, -0.5397942, 0.4862989, -0.9511107, 0.9837515
5: 0.1773891, 1.1126927, 0.0389948, 1.1140633, -0.9366742, 1.0736978
6: -0.3772916, 0.5011861, -0.4470060, 0.6063011, -0.9835927, 0.9481921
7: -0.4012962, 0.4221447, -0.4750039, 0.4184728, -0.8197690, 0.8971485
8: -0.4993078, 0.5927401, -0.6065801, 0.6932982, -1.1926060, 1.1993203
9: -0.4120259, 0.4737192, -0.4302181, 0.5833251, -0.9953510, 0.9039373

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B2_B2_B1_B2_B1

### Relational analysis result of NS_B1_A1_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7455782
time: 1.37 seconds

## Relational analysis of NS_B1_A1_B2_B2_B1_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7387646
time: 1.45 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.5225610, 0.5771261, -0.5361098, 0.5690651, -1.0916262, 1.1132360
1: -0.3550973, 0.3733763, -0.3507358, 0.3569551, -0.7120525, 0.7241121
2: -0.3535787, 0.5432601, -0.3457729, 0.5304011, -0.8839799, 0.8890330
3: -0.3177934, 0.4888853, -0.3141985, 0.5087676, -0.8265610, 0.8030838
4: -0.4917163, 0.4752635, -0.5114608, 0.4571494, -0.9488657, 0.9867243
5: 0.1232671, 1.1177198, 0.1350443, 1.1229123, -0.9996451, 0.9826754
6: -0.4001473, 0.5258820, -0.4000901, 0.5395261, -0.9396734, 0.9259721
7: -0.4258308, 0.4614533, -0.4171859, 0.4705609, -0.8963917, 0.8786392
8: -0.5329626, 0.6193352, -0.5323823, 0.6300368, -1.1629994, 1.1517175
9: -0.4430864, 0.5056530, -0.4463016, 0.4978492, -0.9409356, 0.9519546

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B2_B2_B2_B1_B1

### Relational analysis result of NS_B1_A1_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7509678
time: 1.36 seconds

## Relational analysis of NS_B1_A1_B2_B2_B2_B1_B2

### Relational analysis result of NS_B1_A1_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7425660
time: 1.28 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.4618433, 0.4989081, -0.8770657, 0.8977821, -1.3596255, 1.3759737
1: -0.2976781, 0.3134991, -0.6041457, 0.5977377, -0.8954158, 0.9176447
2: -0.3019718, 0.4503311, -0.5871105, 0.8835375, -1.1855093, 1.0374416
3: -0.2713580, 0.4421914, -0.5087967, 0.8027052, -1.0740632, 0.9509881
4: -0.4311804, 0.4010950, -0.8655260, 0.7864950, -1.2176753, 1.2666210
5: 0.2434835, 1.1058344, -0.3951610, 1.1982327, -0.9547491, 1.5009954
6: -0.3489802, 0.4674866, -0.6424716, 0.8781912, -1.2271714, 1.1099582
7: -0.3728967, 0.3734077, -0.6399112, 0.9102703, -1.2831671, 1.0133189
8: -0.4587702, 0.5561173, -0.8739837, 0.9874951, -1.4462652, 1.4301010
9: -0.3698861, 0.4363161, -0.8297741, 0.7832792, -1.1531652, 1.2660902

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B2_B2_B2_B2_B1

### Relational analysis result of NS_B1_A1_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7338122
time: 1.34 seconds

## Relational analysis of NS_B1_A1_B2_B2_B2_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139773
time: 1.22 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.5130745, 0.5640157, -0.3126657, 0.3390453, -0.8521199, 0.8766814
1: -0.3455592, 0.3641017, -0.1875339, 0.1997154, -0.5452746, 0.5516356
2: -0.3457892, 0.5263857, -0.1873114, 0.2921986, -0.6379878, 0.7136971
3: -0.3097587, 0.4817606, -0.1804033, 0.3046026, -0.6143613, 0.6621639
4: -0.4819413, 0.4640352, -0.2828979, 0.2517706, -0.7337118, 0.7469332
5: 0.1428565, 1.1158948, 0.4840174, 1.0760959, -0.9332395, 0.6318774
6: -0.3918597, 0.5170248, -0.2362811, 0.3138198, -0.7056794, 0.7533059
7: -0.4169897, 0.4472790, -0.2621846, 0.2249296, -0.6419194, 0.7094636
8: -0.5208727, 0.6095918, -0.2975479, 0.3933736, -0.9142463, 0.9071398
9: -0.4318774, 0.4940512, -0.2198371, 0.2975337, -0.7294110, 0.7138883

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B1_A1_A1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7541090, upper bound: 1.7139773
time: 1.57 seconds

## Relational analysis of NS_B1_A2_B1_A1_A1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7490314, upper bound: 1.7139773
time: 1.46 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -1.0030553, 1.1352062, -0.2800578, 0.2997338, -1.3027891, 1.4152640
1: -0.7904509, 0.7956752, -0.1633588, 0.1766350, -0.9670860, 0.9590340
2: -0.6924130, 1.2946717, -0.1635001, 0.2558951, -0.9483081, 1.4581717
3: -0.6562513, 0.8282326, -0.1582504, 0.2703908, -0.9266421, 0.9864830
4: -0.9816868, 0.9482478, -0.2518299, 0.2233040, -1.2049909, 1.2000777
5: -0.7364088, 1.2003856, 0.5380328, 1.0716782, -1.8080870, 0.6623527
6: -0.7685252, 0.9499912, -0.2112061, 0.2783193, -1.0468445, 1.1611973
7: -0.8049885, 1.0839967, -0.2363532, 0.2022176, -1.0072061, 1.3203499
8: -1.0422428, 1.1308882, -0.2601812, 0.3569695, -1.3992124, 1.3910694
9: -0.9352996, 1.0198896, -0.1943015, 0.2630492, -1.1983489, 1.2141911

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B1_A1_A1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7416570, upper bound: 1.7139773
time: 1.47 seconds

## Relational analysis of NS_B1_A2_B1_A1_A1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7355521, upper bound: 1.7139773
time: 1.36 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -1.0838045, 1.0043924, -0.2908989, 0.3129864, -1.3967909, 1.2952913
1: -0.7160311, 0.7594247, -0.1715090, 0.1844152, -0.9004463, 0.9309337
2: -0.7747318, 1.1004987, -0.1715274, 0.2679143, -1.0426461, 1.2720261
3: -0.8665360, 0.7425441, -0.1656470, 0.2819246, -1.1484606, 0.9081911
4: -0.9473826, 0.9365021, -0.2622715, 0.2329008, -1.1802833, 1.1987736
5: -0.5625988, 1.2183427, 0.5198231, 1.0731133, -1.6357121, 0.6985195
6: -0.7456982, 0.8939927, -0.2195167, 0.2902872, -1.0359854, 1.1135094
7: -0.8723270, 0.9703689, -0.2450617, 0.2096527, -1.0819798, 1.2154306
8: -1.0223668, 1.0092521, -0.2727380, 0.3692421, -1.3916090, 1.2819901
9: -0.8962850, 0.9384469, -0.2029101, 0.2746170, -1.1709020, 1.1413569

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B1_A1_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7474594, upper bound: 1.7139773
time: 1.74 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7396925, upper bound: 1.7139773
time: 1.56 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -2.0791819, 1.6615130, -0.2584872, 0.2733644, -2.3525462, 1.9200001
1: -1.2848620, 1.3937132, -0.1471428, 0.1611551, -1.4460171, 1.5408561
2: -1.5280352, 1.9411160, -0.1475281, 0.2319800, -1.7600151, 2.0886440
3: -1.8964746, 1.1290960, -0.1435331, 0.2474426, -2.1439173, 1.2726291
4: -1.7099682, 1.7067119, -0.2310542, 0.2042095, -1.9141777, 1.9377661
5: -1.6338611, 1.3964324, 0.5742651, 1.0688220, -2.7026830, 0.8221673
6: -1.3217535, 1.4898361, -0.1946705, 0.2545065, -1.5762600, 1.6845065
7: -1.6607161, 1.7745793, -0.2190263, 0.1874237, -1.8481398, 1.9936056
8: -1.8392152, 1.6106117, -0.2351973, 0.3325505, -2.1717658, 1.8458090
9: -1.6478782, 1.6250008, -0.1771730, 0.2400331, -1.8879112, 1.8021739

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B1_A1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7322638, upper bound: 1.7139773
time: 1.48 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2_A2_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139215, upper bound: 1.7139773
time: 1.16 seconds

## BFS NS instance: NS_B1_A2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.5824973, 0.6509483, -0.3362445, 0.3670138, -0.9495111, 0.9871928
1: -0.4124871, 0.4279054, -0.2047334, 0.2161381, -0.6286252, 0.6326388
2: -0.3981982, 0.6409445, -0.2042521, 0.3185761, -0.7167743, 0.8451966
3: -0.3625643, 0.5321787, -0.1963432, 0.3289424, -0.6915067, 0.7285219
4: -0.5534573, 0.5378292, -0.3050821, 0.2720228, -0.8254801, 0.8429113
5: 0.0098586, 1.1284109, 0.4455880, 1.0793748, -1.0695162, 0.6828229
6: -0.4485344, 0.5798333, -0.2544780, 0.3390768, -0.7876112, 0.8343113
7: -0.4759852, 0.5426118, -0.2805625, 0.2416429, -0.7176281, 0.8231743
8: -0.6003295, 0.6839519, -0.3242342, 0.4192732, -1.0196027, 1.0081861
9: -0.5073516, 0.5735971, -0.2380043, 0.3222132, -0.8295649, 0.8116013

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B1_A2_A1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7541090, upper bound: 1.7139773
time: 1.60 seconds

## Relational analysis of NS_B1_A2_B1_A2_A1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7490314, upper bound: 1.7139773
time: 1.50 seconds

## BFS NS instance: NS_B1_A2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -1.3676136, 1.2199979, -0.2961439, 0.3193987, -1.6870123, 1.5161419
1: -0.9453129, 0.9719244, -0.1754519, 0.1881792, -1.1334921, 1.1473763
2: -0.9791516, 1.4431726, -0.1754112, 0.2737295, -1.2528811, 1.6185838
3: -1.1674902, 0.9001206, -0.1692258, 0.2875044, -1.4549947, 1.0693464
4: -1.1569054, 1.2286916, -0.2673234, 0.2375436, -1.3944490, 1.4960150
5: -0.9348269, 1.2809613, 0.5110127, 1.0738078, -2.0086346, 0.7699486
6: -0.8778054, 1.1087314, -0.2235374, 0.2960777, -1.1738831, 1.3322688
7: -1.1000483, 1.2482648, -0.2492748, 0.2132500, -1.3132982, 1.4975396
8: -1.2851175, 1.2367293, -0.2788129, 0.3751798, -1.6602973, 1.5155423
9: -1.1451513, 1.1701102, -0.2070750, 0.2802137, -1.4253650, 1.3771852

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B1_A2_A1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7416570, upper bound: 1.7139773
time: 1.43 seconds

## Relational analysis of NS_B1_A2_B1_A2_A1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7355521, upper bound: 1.7139773
time: 1.37 seconds

## BFS NS instance: NS_B1_A2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.9264266, 0.8717961, -0.2471299, 0.2608780, -1.1873047, 1.1189259
1: -0.6290656, 0.6569661, -0.1393407, 0.1530044, -0.7820700, 0.7963067
2: -0.6657203, 0.9306954, -0.1395657, 0.2193884, -0.8851087, 1.0702611
3: -0.7176938, 0.6733429, -0.1357840, 0.2370843, -0.9547781, 0.8091270
4: -0.8399976, 0.8124627, -0.2209985, 0.1941557, -1.0341533, 1.0334612
5: -0.4071755, 1.1940678, 0.5926284, 1.0673183, -1.4744939, 0.6014395
6: -0.6778753, 0.8033594, -0.1864320, 0.2419685, -0.9198439, 0.9897915
7: -0.7222469, 0.8362852, -0.2105286, 0.1796343, -0.9018812, 1.0468138
8: -0.9178644, 0.8884717, -0.2224340, 0.3196936, -1.2375579, 1.1109056
9: -0.7636122, 0.8223966, -0.1681544, 0.2284738, -0.9920859, 0.9905510

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B1_A2_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7571323, upper bound: 1.7415417
time: 1.50 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2_A1_A2

### Relational analysis result of NS_B1_A2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7362933, upper bound: 1.7412851
time: 1.29 seconds

## BFS NS instance: NS_B1_A2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -1.3585025, 1.0921333, -0.2950723, 0.3180887, -1.6765912, 1.3872056
1: -0.8857903, 0.9524533, -0.1746463, 0.1874103, -1.0732006, 1.1270995
2: -1.0249588, 1.2237327, -0.1746177, 0.2725414, -1.2975003, 1.3983505
3: -1.2223724, 0.8284333, -0.1684945, 0.2863645, -1.5087368, 0.9969278
4: -1.2181665, 1.1336007, -0.2662913, 0.2365950, -1.4547615, 1.3998921
5: -0.9030471, 1.3158089, 0.5128128, 1.0736660, -1.9767131, 0.8029961
6: -0.9729307, 1.0938207, -0.2227160, 0.2948946, -1.2678252, 1.3165367
7: -1.0171168, 1.1711167, -0.2484141, 0.2125151, -1.2296319, 1.4195307
8: -1.3316466, 1.1118885, -0.2775718, 0.3739668, -1.7056134, 1.3894603
9: -1.0920794, 1.1093866, -0.2062241, 0.2790701, -1.3711495, 1.3156106

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B1_A2_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7650095, upper bound: 1.7416129
time: 1.46 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7417497, upper bound: 1.7416129
time: 1.23 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.6387188, 0.7107055, -0.3383390, 0.3694990, -1.0082178, 1.0490446
1: -0.4604185, 0.4748269, -0.2062614, 0.2175972, -0.6780157, 0.6810883
2: -0.4390764, 0.7219782, -0.2057571, 0.3209193, -0.7599957, 0.9277354
3: -0.4060808, 0.5686716, -0.1977593, 0.3311050, -0.7371858, 0.7664310
4: -0.6078629, 0.5909225, -0.3070530, 0.2738221, -0.8816850, 0.8979754
5: -0.0824074, 1.1384251, 0.4421737, 1.0796660, -1.1620734, 0.6962514
6: -0.4890341, 0.6265883, -0.2560947, 0.3413207, -0.8303548, 0.8826830
7: -0.5209181, 0.6096226, -0.2821952, 0.2431276, -0.7640458, 0.8918178
8: -0.6569941, 0.7391556, -0.3266051, 0.4215740, -1.0785681, 1.0657606
9: -0.5607138, 0.6306074, -0.2396183, 0.3244056, -0.8851193, 0.8702258

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7538090, upper bound: 1.7135653
time: 1.51 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7485881, upper bound: 1.7135653
time: 1.55 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1.9446411, 1.4282284, -0.3054794, 0.3305213, -2.2751625, 1.7337078
1: -1.2313678, 1.2723289, -0.1822919, 0.1947102, -1.4260780, 1.4546208
2: -1.4570062, 1.6331465, -0.1821483, 0.2841594, -1.7411656, 1.8152947
3: -1.7810998, 1.0027566, -0.1755451, 0.2971843, -2.0782840, 1.1783017
4: -1.6266879, 1.6215624, -0.2761368, 0.2455978, -1.8722857, 1.8976991
5: -1.5367998, 1.3633692, 0.4957299, 1.0750971, -2.6118970, 0.8676394
6: -1.3786309, 1.3808870, -0.2307350, 0.3061222, -1.6847531, 1.6116220
7: -1.4339525, 1.6167941, -0.2565835, 0.2198358, -1.6537883, 1.8733776
8: -1.8300499, 1.3905123, -0.2894148, 0.3854795, -2.2155294, 1.6799271
9: -1.4115245, 1.4875557, -0.2142999, 0.2900124, -1.7015369, 1.7018557

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7407451, upper bound: 1.7135653
time: 1.44 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7350036, upper bound: 1.7135653
time: 1.40 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1.0975589, 0.9655766, -0.2498397, 0.2637384, -1.3612972, 1.2154163
1: -0.7408518, 0.7769970, -0.1411398, 0.1549491, -0.8958009, 0.9181368
2: -0.8177987, 1.0467860, -0.1414275, 0.2223926, -1.0401913, 1.1882135
3: -0.9268211, 0.7357367, -0.1376329, 0.2394095, -1.1662307, 0.8733696
4: -0.9999319, 0.9437956, -0.2233229, 0.1965545, -1.1964865, 1.1671184
5: -0.6110394, 1.2439737, 0.5883076, 1.0676771, -1.6787165, 0.6556661
6: -0.7972164, 0.9308761, -0.1883580, 0.2449597, -1.0421761, 1.1192341
7: -0.8477933, 0.9720467, -0.2125030, 0.1814929, -1.0292861, 1.1845498
8: -1.0901942, 0.9792317, -0.2254460, 0.3227613, -1.4129555, 1.2046777
9: -0.9012591, 0.9370662, -0.1703061, 0.2311843, -1.1324434, 1.1073723

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7571030, upper bound: 1.7415417
time: 1.49 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7362933, upper bound: 1.7412851
time: 1.29 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1.5256275, 1.2125639, -0.2950946, 0.3181155, -1.8437431, 1.5076585
1: -1.0302637, 1.0900173, -0.1746630, 0.1874261, -1.2176898, 1.2646804
2: -1.2110850, 1.3374087, -0.1746341, 0.2725661, -1.4836510, 1.5120428
3: -1.4603341, 0.8968014, -0.1685098, 0.2863881, -1.7467222, 1.0653112
4: -1.4092669, 1.2832818, -0.2663126, 0.2366147, -1.6458817, 1.5495944
5: -1.1271586, 1.3766431, 0.5127756, 1.0736687, -2.2008274, 0.8638675
6: -1.1015533, 1.2677469, -0.2227329, 0.2949192, -1.3964725, 1.4904798
7: -1.1788001, 1.3198433, -0.2484319, 0.2125303, -1.3913304, 1.5682752
8: -1.5264699, 1.2122338, -0.2775976, 0.3739917, -1.9004617, 1.4898314
9: -1.2536669, 1.2283387, -0.2062417, 0.2790938, -1.5327607, 1.4345804

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7651389, upper bound: 1.7416129
time: 1.45 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7417497, upper bound: 1.7416129
time: 1.18 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.7119803, 0.7588112, -0.3850430, 0.4232965, -1.1352768, 1.1438541
1: -0.5038248, 0.5226871, -0.2390265, 0.2501809, -0.7540057, 0.7617136
2: -0.4932940, 0.7857556, -0.2391254, 0.3738037, -0.8670977, 1.0248811
3: -0.4814702, 0.5985870, -0.2280319, 0.3784219, -0.8598920, 0.8266189
4: -0.6633381, 0.6490328, -0.3544517, 0.3122852, -0.9756233, 1.0034846
5: -0.1636878, 1.1518843, 0.3691384, 1.0895119, -1.2531997, 0.7827460
6: -0.5318309, 0.6718025, -0.2926069, 0.3896344, -0.9214653, 0.9644094
7: -0.5767838, 0.6710039, -0.3184187, 0.2778150, -0.8545988, 0.9894226
8: -0.7180949, 0.7841443, -0.3808998, 0.4717198, -1.1898148, 1.1650442
9: -0.6150457, 0.6823090, -0.2773459, 0.3712761, -0.9863218, 0.9596549

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B2_B2_A1_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7547507, upper bound: 1.7139773
time: 1.35 seconds

## Relational analysis of NS_B1_A2_B2_B2_A1_A1_A2

### Relational analysis result of NS_B1_A2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7492547, upper bound: 1.7139773
time: 1.55 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -2.1611617, 1.5028157, -0.3510554, 0.3846465, -2.5458081, 1.8538711
1: -1.3649787, 1.4506590, -0.2155371, 0.2264809, -1.5914596, 1.6661961
2: -1.6897349, 1.7340899, -0.2149960, 0.3351450, -2.0248799, 1.9490860
3: -2.0754347, 1.1179150, -0.2063557, 0.3442794, -2.4197140, 1.3242707
4: -1.8922721, 1.7330177, -0.3191629, 0.2847444, -2.1770165, 2.0521805
5: -1.8205051, 1.4711586, 0.4214485, 1.0816169, -2.9021220, 1.0497102
6: -1.5106299, 1.5862211, -0.2660258, 0.3549419, -1.8655717, 1.8522469
7: -1.5538585, 1.7832650, -0.2922309, 0.2521413, -1.8059998, 2.0754960
8: -2.1039896, 1.4650203, -0.3411805, 0.4355421, -2.5395317, 1.8062007
9: -1.6374397, 1.6352855, -0.2494651, 0.3377155, -1.9751552, 1.8847506

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A2_B2_B2_A1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7421488, upper bound: 1.7139773
time: 1.41 seconds

## Relational analysis of NS_B1_A2_B2_B2_A1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7357030, upper bound: 1.7139773
time: 1.31 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1.2128069, 1.0320723, -0.2926058, 0.3150730, -1.5278800, 1.3246781
1: -0.8187695, 0.8612705, -0.1727919, 0.1856401, -1.0044096, 1.0340625
2: -0.9236822, 1.1250302, -0.1727912, 0.2698069, -1.1934891, 1.2978213
3: -1.0704572, 0.7790995, -0.1668115, 0.2837403, -1.3541975, 0.9459110
4: -1.1101359, 1.0351940, -0.2639154, 0.2344116, -1.3445475, 1.2991095
5: -0.7499927, 1.2796917, 0.5169560, 1.0733393, -1.8233321, 0.7627357
6: -0.8791523, 1.0215707, -0.2208251, 0.2921715, -1.1713238, 1.2423959
7: -0.9369088, 1.0656831, -0.2464327, 0.2108233, -1.1477320, 1.3121158
8: -1.2076516, 1.0419616, -0.2747149, 0.3711742, -1.5788257, 1.3166765
9: -0.9961368, 1.0154848, -0.2042654, 0.2764382, -1.2725750, 1.2197502

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7571895, upper bound: 1.7415417
time: 1.54 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7362933, upper bound: 1.7412851
time: 1.29 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.6442757, 1.2810209, -0.3403140, 0.3718412, -2.0161169, 1.6213349
1: -1.1104805, 1.1767770, -0.2077018, 0.2189725, -1.3294530, 1.3844788
2: -1.3200920, 1.4179600, -0.2071759, 0.3231286, -1.6432207, 1.6251360
3: -1.6082077, 0.9414437, -0.1990942, 0.3331434, -1.9413512, 1.1405380
4: -1.5227215, 1.3773774, -0.3089109, 0.2755181, -1.7982397, 1.6862884
5: -1.2702115, 1.4134147, 0.4389552, 1.0799407, -2.3501520, 0.9744595
6: -1.1859062, 1.3611169, -0.2576186, 0.3434361, -1.5293422, 1.6187356
7: -1.2705450, 1.4162419, -0.2837343, 0.2445274, -1.5150723, 1.6999762
8: -1.6473924, 1.2768148, -0.3288401, 0.4237433, -2.0711358, 1.6056550
9: -1.3513434, 1.3090707, -0.2411397, 0.3264726, -1.6778159, 1.5502105

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7651663, upper bound: 1.7416129
time: 1.47 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7417497, upper bound: 1.7416129
time: 1.16 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.3068123, 0.3321023, -0.3704858, 0.4077095, -0.7145218, 0.7025881
1: -0.1832643, 0.1956385, -0.2295428, 0.2399622, -0.4232265, 0.4251813
2: -0.1831059, 0.2856508, -0.2293322, 0.3575211, -0.5406270, 0.5149829
3: -0.1764463, 0.2985602, -0.2192903, 0.3645611, -0.5410074, 0.5178505
4: -0.2773910, 0.2467429, -0.3389910, 0.3011785, -0.5785695, 0.5857339
5: 0.4935573, 1.0752822, 0.3902645, 1.0860518, -0.5924945, 0.6850178
6: -0.2317638, 0.3075498, -0.2814914, 0.3754368, -0.6072006, 0.5890411
7: -0.2576224, 0.2207806, -0.3078576, 0.2666885, -0.5243109, 0.5286382
8: -0.2909233, 0.3869441, -0.3647020, 0.4565584, -0.7474817, 0.7516462
9: -0.2153271, 0.2914075, -0.2649941, 0.3577417, -0.5730687, 0.5564016

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_B1

### Relational analysis result of NS_B2_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7814949
time: 1.36 seconds

## Relational analysis of NS_B2_A1_B1_A1_B1_B1_B2

### Relational analysis result of NS_B2_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7679741
time: 1.30 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.2852308, 0.3060575, -0.6133206, 0.6583737, -0.9436045, 0.9193780
1: -0.1672477, 0.1803475, -0.4180177, 0.4517237, -0.6189715, 0.5983652
2: -0.1673304, 0.2616305, -0.4265762, 0.6028301, -0.7701604, 0.6882067
3: -0.1617797, 0.2758944, -0.3559933, 0.5694093, -0.7311890, 0.6318877
4: -0.2568123, 0.2278831, -0.5840803, 0.5818733, -0.8386856, 0.8119634
5: 0.5293438, 1.0723630, -0.0062164, 1.1383607, -0.6090169, 1.0785794
6: -0.2151716, 0.2840299, -0.4658244, 0.6255086, -0.8406802, 0.7498543
7: -0.2405086, 0.2057653, -0.4908034, 0.5607468, -0.8012555, 0.6965687
8: -0.2661728, 0.3628254, -0.6216654, 0.7271054, -0.9932783, 0.9844909
9: -0.1984092, 0.2685689, -0.5680279, 0.5563917, -0.7548009, 0.8365967

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_B1

### Relational analysis result of NS_B2_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740677
time: 1.33 seconds

## Relational analysis of NS_B2_A1_B1_A1_B1_B2_B2

### Relational analysis result of NS_B2_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7513879
time: 1.20 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2930965, 0.3156731, -0.4926591, 0.5356207, -0.8287172, 0.8083322
1: -0.1731610, 0.1859921, -0.3254646, 0.3439290, -0.5170899, 0.5114567
2: -0.1731547, 0.2703509, -0.3289933, 0.4897451, -0.6628997, 0.5993442
3: -0.1671464, 0.2842625, -0.2922832, 0.4664456, -0.6335920, 0.5765457
4: -0.2643881, 0.2348459, -0.4611439, 0.4396134, -0.7040015, 0.6959898
5: 0.5161319, 1.0734043, 0.1848604, 1.1120099, -0.5958780, 0.8885440
6: -0.2212012, 0.2927135, -0.3741442, 0.4977597, -0.7189609, 0.6668577
7: -0.2468269, 0.2111600, -0.3979007, 0.4167413, -0.6635683, 0.6090606
8: -0.2752834, 0.3717296, -0.4946511, 0.5890943, -0.8643777, 0.8663808
9: -0.2046551, 0.2769617, -0.4077625, 0.4693204, -0.6739755, 0.6847242

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_B2_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7843712, upper bound: 1.7513879
time: 1.41 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_B2_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7703235, upper bound: 1.7513879
time: 2.52 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4816105, 0.5394434, -0.4585243, 0.4955992, -0.9772097, 0.9979677
1: -0.3107700, 0.3173861, -0.2951813, 0.3102477, -0.6210176, 0.6125674
2: -0.3086933, 0.4811963, -0.2990189, 0.4471670, -0.7558602, 0.7802151
3: -0.2946143, 0.4790012, -0.2694645, 0.4395516, -0.7341659, 0.7484657
4: -0.4418494, 0.3968819, -0.4280078, 0.3969244, -0.8387738, 0.8248897
5: 0.2086653, 1.0995879, 0.2492833, 1.1051594, -0.8964941, 0.8503046
6: -0.3666645, 0.4947889, -0.3464362, 0.4642078, -0.8308724, 0.8412251
7: -0.3938640, 0.3446821, -0.3704502, 0.3690485, -0.7629125, 0.7151324
8: -0.4887579, 0.5789480, -0.4553903, 0.5525134, -1.0412713, 1.0343382
9: -0.3500077, 0.4743637, -0.3657749, 0.4332337, -0.7832414, 0.8401386

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_B2_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7763739, upper bound: 1.7513879
time: 1.52 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_B2_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7513879
time: 1.27 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.2931429, 0.3157300, -0.5358343, 0.5956620, -0.8888049, 0.8515642
1: -0.1731959, 0.1860255, -0.3694362, 0.3863520, -0.5595480, 0.5554618
2: -0.1731890, 0.2704023, -0.3647101, 0.5668687, -0.7400577, 0.6351124
3: -0.1671781, 0.2843120, -0.3290347, 0.4989328, -0.6661109, 0.6133468
4: -0.2644329, 0.2348872, -0.5056198, 0.4909726, -0.7554055, 0.7405071
5: 0.5160534, 1.0734106, 0.0949022, 1.1202732, -0.6042198, 0.9785085
6: -0.2212371, 0.2927647, -0.4121744, 0.5382746, -0.7595117, 0.7049391
7: -0.2468642, 0.2111918, -0.4384236, 0.4812849, -0.7281491, 0.6496154
8: -0.2753370, 0.3717824, -0.5498775, 0.6340712, -0.9094083, 0.9216599
9: -0.2046920, 0.2770114, -0.4587692, 0.5226846, -0.7273766, 0.7357806

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B1_A2_A1_A1_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7843712, upper bound: 1.7513879
time: 1.48 seconds

## Relational analysis of NS_B2_A1_B1_A2_A1_A1_A2

### Relational analysis result of NS_B2_A1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7702354, upper bound: 1.7513879
time: 1.39 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.4616129, 0.5157229, -0.4989119, 0.5443690, -1.0059819, 1.0146347
1: -0.2961828, 0.3034576, -0.3316513, 0.3501443, -0.6463271, 0.6351089
2: -0.2943257, 0.4588248, -0.3341681, 0.5010228, -0.7953485, 0.7929929
3: -0.2810954, 0.4583581, -0.2976673, 0.4711417, -0.7522371, 0.7560254
4: -0.4230347, 0.3797054, -0.4674970, 0.4471377, -0.8701724, 0.8472024
5: 0.2412581, 1.0968072, 0.1719193, 1.1131922, -0.8719342, 0.9248880
6: -0.3512313, 0.4733684, -0.3795956, 0.5036952, -0.8549265, 0.8529640
7: -0.3782773, 0.3305074, -0.4037819, 0.4261003, -0.8043776, 0.7342893
8: -0.4661249, 0.5569820, -0.5027172, 0.5954095, -1.0615344, 1.0596993
9: -0.3345997, 0.4534325, -0.4151472, 0.4769399, -0.8115396, 0.8685797

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B1_A2_A1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7763739, upper bound: 1.7513879
time: 1.41 seconds

## Relational analysis of NS_B2_A1_B1_A2_A1_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7513879
time: 1.29 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.3390830, 0.3703811, -0.5721067, 0.6389837, -0.9780667, 0.9424878
1: -0.2068040, 0.2181151, -0.4031490, 0.4188189, -0.6256229, 0.6212641
2: -0.2062916, 0.3217516, -0.3909290, 0.6247930, -0.8310845, 0.7126806
3: -0.1982621, 0.3318728, -0.3553083, 0.5248644, -0.7231265, 0.6871810
4: -0.3077528, 0.2744609, -0.5428767, 0.5276891, -0.8354419, 0.8173376
5: 0.4409614, 1.0797694, 0.0282960, 1.1266326, -0.6856712, 1.0514734
6: -0.2566687, 0.3421173, -0.4406285, 0.5706877, -0.8273563, 0.7827458
7: -0.2827750, 0.2436551, -0.4678566, 0.5292360, -0.8120109, 0.7115116
8: -0.3274469, 0.4223912, -0.5894111, 0.6729095, -1.0003564, 1.0118024
9: -0.2401915, 0.3251842, -0.4967784, 0.5625704, -0.8027619, 0.8219626

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B1_A2_A2_A1_A1

### Relational analysis result of NS_B2_A1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7843712, upper bound: 1.7513879
time: 1.55 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2_A1_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7702354, upper bound: 1.7513879
time: 1.53 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.5677752, 0.5995942, -0.5302241, 0.5878270, -1.1556022, 1.1298182
1: -0.3742709, 0.3793174, -0.3633756, 0.3808677, -0.7551386, 0.7426929
2: -0.3681867, 0.5631977, -0.3600051, 0.5568899, -0.9250766, 0.9232029
3: -0.3322715, 0.5360665, -0.3242833, 0.4946861, -0.8269576, 0.8603498
4: -0.5443439, 0.4877368, -0.4997432, 0.4843328, -1.0286767, 0.9874799
5: 0.0858025, 1.1299075, 0.1068912, 1.1191941, -1.0333917, 1.0230162
6: -0.4226010, 0.5709791, -0.4070909, 0.5330365, -0.9556375, 0.9780699
7: -0.4378710, 0.5113983, -0.4331010, 0.4729026, -0.9107735, 0.9444993
8: -0.5641079, 0.6632349, -0.5427278, 0.6278428, -1.1919507, 1.2059628
9: -0.4819160, 0.5243580, -0.4521405, 0.5154858, -0.9974018, 0.9764986

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B1_A2_A2_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7763739, upper bound: 1.7513879
time: 1.45 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7513879
time: 1.24 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.3126657, 0.3390453, -0.5604095, 0.6255153, -0.9381810, 0.8994549
1: -0.1875339, 0.1997154, -0.3926367, 0.4085899, -0.5961238, 0.5923520
2: -0.1873114, 0.2921986, -0.3827460, 0.6066110, -0.7939224, 0.6749446
3: -0.1804033, 0.3046026, -0.3471398, 0.5166302, -0.6970334, 0.6517425
4: -0.2828979, 0.2517706, -0.5309665, 0.5162743, -0.7991723, 0.7827371
5: 0.4840174, 1.0760959, 0.0490520, 1.1246309, -0.6406135, 1.0270439
6: -0.2362811, 0.3138198, -0.4317286, 0.5603929, -0.7966740, 0.7455484
7: -0.2621846, 0.2249296, -0.4587060, 0.5141786, -0.7763631, 0.6836357
8: -0.2975479, 0.3933736, -0.5771204, 0.6604789, -0.9580269, 0.9704940
9: -0.2198371, 0.2975337, -0.4848762, 0.5501575, -0.7699946, 0.7824098

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7813746
time: 1.39 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_B1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7678785
time: 1.21 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.2908989, 0.3129864, -1.1900680, 1.0853772, -1.3762761, 1.5030544
1: -0.1715090, 0.1844152, -0.7767578, 0.8306918, -1.0022007, 0.9611731
2: -0.1715274, 0.2679143, -0.8551523, 1.1917607, -1.3632880, 1.1230667
3: -0.1656470, 0.2819246, -0.9764891, 0.7884042, -0.9540511, 1.2584138
4: -0.2622715, 0.2329008, -1.0349638, 1.0187275, -1.2809991, 1.2678646
5: 0.5198231, 1.0731133, -0.6870611, 1.2373548, -0.7175316, 1.7601745
6: -0.2195167, 0.2902872, -0.8071958, 0.9631428, -1.1826595, 1.0974829
7: -0.2450617, 0.2096527, -0.9564929, 1.0631747, -1.3082365, 1.1661456
8: -0.2727380, 0.3692421, -1.1095709, 1.0782899, -1.3510278, 1.4788129
9: -0.2029101, 0.2746170, -0.9768620, 1.0117415, -1.2146516, 1.2514790

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7738816
time: 3.29 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_B2_B2

### Relational analysis result of NS_B2_A1_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.25 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.3362445, 0.3670138, -0.6615353, 0.7261221, -1.0623666, 1.0285491
1: -0.2047334, 0.2161381, -0.4747416, 0.4903403, -0.6950737, 0.6908797
2: -0.2042521, 0.3185761, -0.4554437, 0.7438841, -0.9481362, 0.7740198
3: -0.1963432, 0.3289424, -0.4297023, 0.5785574, -0.7749006, 0.7586447
4: -0.3050821, 0.2720228, -0.6243479, 0.6097564, -0.9148384, 0.8963708
5: 0.4455880, 1.0793748, -0.1083123, 1.1427915, -0.6972035, 1.1876872
6: -0.2544780, 0.3390768, -0.5015886, 0.6412182, -0.8956962, 0.8406653
7: -0.2805625, 0.2416429, -0.5382924, 0.6291987, -0.9097612, 0.7799354
8: -0.3242342, 0.4192732, -0.6760284, 0.7541374, -1.0783716, 1.0953016
9: -0.2380043, 0.3222132, -0.5777697, 0.6471903, -0.8851946, 0.8999829

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7813746
time: 1.35 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_B1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7678785
time: 1.33 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.3144281, 0.3411362, -1.7736273, 1.3038270, -1.6182551, 2.1147635
1: -0.1888195, 0.2009431, -1.1324434, 1.2363479, -1.4251674, 1.3333865
2: -0.1885778, 0.2941701, -1.3701042, 1.5052747, -1.6938524, 1.6642742
3: -0.1815948, 0.3064219, -1.7072520, 0.9774384, -1.1590332, 2.0136740
4: -0.2845563, 0.2532840, -1.5814997, 1.4421396, -1.7266960, 1.8347838
5: 0.4811449, 1.0763413, -1.3794663, 1.4327736, -0.9516287, 2.4558077
6: -0.2376412, 0.3157079, -1.2564107, 1.3728870, -1.6105282, 1.5721186
7: -0.2635584, 0.2261789, -1.3004181, 1.4928133, -1.7563717, 1.5265970
8: -0.2995428, 0.3953093, -1.7291960, 1.3265409, -1.6260837, 2.1245053
9: -0.2211950, 0.2993786, -1.4076599, 1.3851180, -1.6063130, 1.7070384

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B2_A1_B2_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7903640, upper bound: 1.7861047
time: 1.59 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
time: 1.36 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3383390, 0.3694990, -0.7854890, 0.7998929, -1.1382320, 1.1549881
1: -0.2062614, 0.2175972, -0.5462473, 0.5683002, -0.7745616, 0.7638445
2: -0.2057571, 0.3209193, -0.5502546, 0.8379964, -1.0437535, 0.8711739
3: -0.1977593, 0.3311050, -0.5595360, 0.6238066, -0.8215659, 0.8906410
4: -0.3070530, 0.2738221, -0.7212454, 0.7065541, -1.0136070, 0.9950675
5: 0.4421737, 1.0796660, -0.2467588, 1.1641203, -0.7219466, 1.3264248
6: -0.2560947, 0.3413207, -0.5832387, 0.7148548, -0.9709495, 0.9245594
7: -0.2821952, 0.2431276, -0.6268544, 0.7297107, -1.0119059, 0.8699821
8: -0.3266051, 0.4215740, -0.7848384, 0.8209969, -1.1476020, 1.2064124
9: -0.2396183, 0.3244056, -0.6645221, 0.7313124, -0.9709307, 0.9889276

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7812601
time: 1.38 seconds

## Relational analysis of NS_B2_A1_B2_A2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7675506
time: 1.33 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3138034, 0.3403951, -1.9489357, 1.4543340, -1.7681375, 2.2893310
1: -0.1883638, 0.2005080, -1.3162967, 1.3964250, -1.5847888, 1.5168047
2: -0.1881289, 0.2934714, -1.5990731, 1.6218921, -1.8100210, 1.8925444
3: -0.1811725, 0.3057770, -1.9855920, 1.0544636, -1.2356361, 2.2913690
4: -0.2839686, 0.2527475, -1.8211017, 1.6155978, -1.8995664, 2.0738492
5: 0.4821630, 1.0762542, -1.6397218, 1.5065098, -1.0243468, 2.7159760
6: -0.2371591, 0.3150386, -1.4147823, 1.5975008, -1.8346599, 1.7298208
7: -0.2630714, 0.2257362, -1.5028149, 1.6604073, -1.9234787, 1.7285510
8: -0.2988358, 0.3946233, -1.9585302, 1.4403136, -1.7391493, 2.3531535
9: -0.2207136, 0.2987248, -1.5986304, 1.5149469, -1.7356606, 1.8973553

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B2_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7903640, upper bound: 1.7861047
time: 1.55 seconds

## Relational analysis of NS_B2_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
time: 1.39 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3850430, 0.4232965, -0.9011389, 0.8589003, -1.2439432, 1.3244354
1: -0.2390265, 0.2501809, -0.6140400, 0.6396728, -0.8786993, 0.8642209
2: -0.2391254, 0.3738037, -0.6446952, 0.9135461, -1.1526716, 1.0184989
3: -0.2280319, 0.3784219, -0.6881574, 0.6642659, -0.8922978, 1.0665793
4: -0.3544517, 0.3122852, -0.8178649, 0.7936682, -1.1481199, 1.1301501
5: 0.3691384, 1.0895119, -0.3781537, 1.1869439, -0.8178055, 1.4676656
6: -0.2926069, 0.3896344, -0.6606063, 0.7863606, -1.0789675, 1.0502408
7: -0.3184187, 0.2778150, -0.7049896, 0.8166884, -1.1351070, 0.9828046
8: -0.3808998, 0.4717198, -0.8936470, 0.8753968, -1.2562966, 1.3653667
9: -0.2773459, 0.3712761, -0.7443895, 0.8055996, -1.0829455, 1.1156657

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_B2_A1_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7814618
time: 1.31 seconds

## Relational analysis of NS_B2_A1_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7678959
time: 1.34 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3602495, 0.3956605, -2.0687399, 1.5231080, -1.8833575, 2.4644003
1: -0.2222115, 0.2329190, -1.3972721, 1.4835864, -1.7057979, 1.6301911
2: -0.2218424, 0.3455530, -1.7090117, 1.7028162, -1.9246587, 2.0545647
3: -0.2125327, 0.3538914, -2.1345775, 1.0993128, -1.3118454, 2.4884689
4: -0.3283492, 0.2925924, -1.9366637, 1.7101288, -2.0384779, 2.2292562
5: 0.4065565, 1.0835268, -1.7844777, 1.5434517, -1.1368952, 2.8680046
6: -0.2734039, 0.3647295, -1.5016984, 1.6913033, -1.9647071, 1.8664279
7: -0.2996936, 0.2588067, -1.5949844, 1.7572680, -2.0569615, 1.8537911
8: -0.3521806, 0.4455785, -2.0807214, 1.5051934, -1.8573740, 2.5263000
9: -0.2567365, 0.3472790, -1.6967585, 1.5962636, -1.8530002, 2.0440376

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B2_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7903640, upper bound: 1.7861047
time: 1.55 seconds

## Relational analysis of NS_B2_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
time: 1.38 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.3213277, 0.3493201, -0.5967087, 0.6673121, -0.9886398, 0.9460288
1: -0.1938525, 0.2057487, -0.4252592, 0.4403328, -0.6341853, 0.6310079
2: -0.1935349, 0.3018887, -0.4081400, 0.6630358, -0.8565706, 0.7100288
3: -0.1862591, 0.3135443, -0.3724883, 0.5421829, -0.7284420, 0.6860326
4: -0.2910477, 0.2592105, -0.5679274, 0.5516981, -0.8427458, 0.8271378
5: 0.4698996, 1.0773007, -0.0153594, 1.1308432, -0.6609436, 1.0926601
6: -0.2429660, 0.3230983, -0.4593475, 0.5923415, -0.8353075, 0.7824458
7: -0.2689360, 0.2310695, -0.4871031, 0.5609061, -0.8298421, 0.7181727
8: -0.3073516, 0.4028880, -0.6152625, 0.6990544, -1.0064061, 1.0181506
9: -0.2265110, 0.3066003, -0.5218129, 0.5886777, -0.8151886, 0.8284132

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of NS_B2_A2_A1_B1_A1_A1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.37 seconds

## Relational analysis of NS_B2_A2_A1_B1_A1_A1_A2

### Relational analysis result of NS_B2_A2_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.21 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.4194033, 0.4558918, -0.6172068, 0.6902741, -1.1096774, 1.0730987
1: -0.2649211, 0.2740566, -0.4435188, 0.4577711, -0.7226923, 0.7175754
2: -0.2655452, 0.4088201, -0.4226125, 0.6940332, -0.9595785, 0.8314326
3: -0.2473766, 0.4075684, -0.3870139, 0.5562207, -0.8035973, 0.7945822
4: -0.3895604, 0.3490234, -0.5886064, 0.5711877, -0.9607482, 0.9376297
5: 0.3155125, 1.0969805, -0.0507447, 1.1343153, -0.8188028, 1.1477251
6: -0.3166411, 0.4244723, -0.4747374, 0.6098932, -0.9265343, 0.8992097
7: -0.3411733, 0.3191743, -0.5033144, 0.5865767, -0.9277500, 0.8224887
8: -0.4147728, 0.5091894, -0.6364334, 0.7202463, -1.1350191, 1.1456227
9: -0.3163287, 0.3995789, -0.5421045, 0.6103253, -0.9266540, 0.9416834

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of NS_B2_A2_A1_B1_A1_A2_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.30 seconds

## Relational analysis of NS_B2_A2_A1_B1_A1_A2_A2

### Relational analysis result of NS_B2_A2_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.21 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.4224386, 0.4587182, -0.5942776, 0.6645132, -1.0869517, 1.0529959
1: -0.2673172, 0.2761270, -0.4230743, 0.4382072, -0.7055244, 0.6992013
2: -0.2680547, 0.4118569, -0.4064397, 0.6592563, -0.9273110, 0.8182966
3: -0.2490765, 0.4100958, -0.3707911, 0.5404715, -0.7895480, 0.7808869
4: -0.3926049, 0.3526539, -0.5654523, 0.5493256, -0.9419305, 0.9181063
5: 0.3106385, 1.0976281, -0.0110456, 1.1304270, -0.8197885, 1.1086737
6: -0.3187253, 0.4276187, -0.4574977, 0.5902021, -0.9089274, 0.8851164
7: -0.3431899, 0.3229553, -0.4852011, 0.5577766, -0.9009665, 0.8081564
8: -0.4177099, 0.5126002, -0.6127084, 0.6964709, -1.1141808, 1.1253085
9: -0.3199375, 0.4020333, -0.5193389, 0.5860981, -0.9060356, 0.9213722

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_B2_A2_A1_B1_A2_A1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7997147, upper bound: 1.8015053
time: 1.54 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2_A1_A2

### Relational analysis result of NS_B2_A2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7996038, upper bound: 1.8015053
time: 1.82 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.5016952, 0.5482630, -0.6245046, 0.6980408, -1.1997360, 1.1727676
1: -0.3344052, 0.3529107, -0.4499033, 0.4636694, -0.7980746, 0.8028140
2: -0.3364717, 0.5060431, -0.4278464, 0.7045176, -1.0409893, 0.9338895
3: -0.3000640, 0.4732320, -0.3923191, 0.5609688, -0.8610328, 0.8655512
4: -0.4703251, 0.4504872, -0.5958532, 0.5777935, -1.0481186, 1.0463405
5: 0.1661586, 1.1137189, -0.0627134, 1.1355257, -0.9693671, 1.1764324
6: -0.3820223, 0.5063373, -0.4800499, 0.6158296, -0.9978519, 0.9863873
7: -0.4063996, 0.4302663, -0.5092077, 0.5952595, -1.0016592, 0.9394740
8: -0.5063076, 0.5982207, -0.6437379, 0.7274144, -1.2337220, 1.2419586
9: -0.4184343, 0.4803315, -0.5489680, 0.6179321, -1.0363665, 1.0292994

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_B2_A2_A1_B1_A2_A2_A1

### Relational analysis result of NS_B2_A2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015216, upper bound: 1.8015053
time: 1.54 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2_A2_A2

### Relational analysis result of NS_B2_A2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015053, upper bound: 1.8015053
time: 1.34 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.3213277, 0.3493201, -1.2146020, 1.0271147, -1.3484424, 1.5639220
1: -0.1938525, 0.2057487, -0.8188346, 0.8558514, -1.0497038, 1.0245833
2: -0.1935349, 0.3018887, -0.9209672, 1.1258385, -1.3193734, 1.2228558
3: -0.1862591, 0.3135443, -1.0691961, 0.7768787, -0.9631377, 1.3827404
4: -0.2910477, 0.2592105, -1.1076713, 1.0305948, -1.3216425, 1.3668817
5: 0.4698996, 1.0773007, -0.7493849, 1.2754468, -0.8055472, 1.8266855
6: -0.2429660, 0.3230983, -0.8764555, 1.0181379, -1.2611039, 1.1995538
7: -0.2689360, 0.2310695, -0.9315189, 1.0616488, -1.3305849, 1.1625885
8: -0.3073516, 0.4028880, -1.2064799, 1.0413868, -1.3487384, 1.6093680
9: -0.2265110, 0.3066003, -0.9941610, 1.0148058, -1.2413168, 1.3007613

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_B2_A2_A1_B2_A1_A1_B1

### Relational analysis result of NS_B2_A2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.25 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.34 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.4194033, 0.4558918, -1.2763524, 1.0606933, -1.4800966, 1.7322443
1: -0.2649211, 0.2740566, -0.8602211, 0.8985052, -1.1634263, 1.1342777
2: -0.2655452, 0.4088201, -0.9761310, 1.1676404, -1.4331857, 1.3849511
3: -0.2473766, 0.4075684, -1.1448598, 0.7990615, -1.0464381, 1.5524282
4: -0.3895604, 0.3490234, -1.1652584, 1.0775504, -1.4671109, 1.5142817
5: 0.3155125, 1.0969805, -0.8229069, 1.2928942, -0.9773818, 1.9198873
6: -0.3166411, 0.4244723, -0.9190297, 1.0648208, -1.3814619, 1.3435020
7: -0.3411733, 0.3191743, -0.9768150, 1.1100001, -1.4511734, 1.2959893
8: -0.4147728, 0.5091894, -1.2684042, 1.0743499, -1.4891226, 1.7775936
9: -0.3163287, 0.3995789, -1.0437621, 1.0560607, -1.3723893, 1.4433410

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_B2_A2_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.43 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.26 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.5522239, 0.6160899, -0.7303391, 0.7696322, -1.3218560, 1.3464290
1: -0.3852801, 0.4014319, -0.5144022, 0.5342680, -0.9195482, 0.9158340
2: -0.3770197, 0.5938873, -0.5074266, 0.7994224, -1.1764421, 1.1013138
3: -0.3414238, 0.5108677, -0.5012291, 0.6053761, -0.9467999, 1.0120969
4: -0.5226315, 0.5082861, -0.6779469, 0.6634290, -1.1860605, 1.1862330
5: 0.0635770, 1.1232300, -0.1845481, 1.1549166, -1.0913396, 1.3077781
6: -0.4255004, 0.5531882, -0.5449648, 0.6826361, -1.1081365, 1.0981530
7: -0.4523025, 0.5036410, -0.5892066, 0.6865842, -1.1388867, 1.0928476
8: -0.5685194, 0.6517796, -0.7346758, 0.7937900, -1.3623095, 1.3864553
9: -0.4765466, 0.5414714, -0.6281897, 0.6950313, -1.1715779, 1.1696610

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_B2_A2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015215, upper bound: 1.7994943
time: 1.67 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015053, upper bound: 1.7994943
time: 1.66 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5812300, 0.6494889, -1.0225120, 0.9226595, -1.5038896, 1.6720009
1: -0.4113484, 0.4267971, -0.6900922, 0.7231655, -1.1345140, 1.1168892
2: -0.3973115, 0.6389747, -0.7493664, 0.9958051, -1.3931166, 1.3883412
3: -0.3616793, 0.5312867, -0.8338242, 0.7078733, -1.0695527, 1.3651109
4: -0.5521665, 0.5365927, -0.9285325, 0.8845283, -1.4366949, 1.4651251
5: 0.0121073, 1.1281939, -0.5206762, 1.2211725, -1.2090652, 1.6488700
6: -0.4475703, 0.5787177, -0.7440170, 0.8729190, -1.3204893, 1.3227347
7: -0.4749938, 0.5409802, -0.7906154, 0.9112385, -1.3862323, 1.3315957
8: -0.5989981, 0.6826048, -1.0138474, 0.9388489, -1.5378469, 1.6964521
9: -0.5060623, 0.5722522, -0.8398656, 0.8864730, -1.3925352, 1.4121177

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of NS_B2_A2_A1_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015053, upper bound: 1.8014497
time: 1.42 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015053, upper bound: 1.8014497
time: 1.47 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -1.2158616, 1.0335090, -0.3213277, 0.3493201, -1.5651816, 1.3548367
1: -0.8208137, 0.8630909, -0.1938525, 0.2057487, -1.0265623, 1.0569434
2: -0.9263667, 1.1267198, -0.1935349, 0.3018887, -1.2282555, 1.3202547
3: -1.0739582, 0.7800362, -0.1862591, 0.3135443, -1.3875024, 0.9662953
4: -1.1139888, 1.0371690, -0.2910477, 0.2592105, -1.3731992, 1.3282167
5: -0.7539648, 1.2804635, 0.4698996, 1.0773007, -1.8312654, 0.8105639
6: -0.8829459, 1.0235298, -0.2429660, 0.3230983, -1.2060442, 1.2664957
7: -0.9388340, 1.0677215, -0.2689360, 0.2310695, -1.1699035, 1.3366575
8: -1.2108495, 1.0433171, -0.3073516, 0.4028880, -1.6137376, 1.3506687
9: -0.9981867, 1.0173749, -0.2265110, 0.3066003, -1.3047870, 1.2438859

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of NS_B2_A2_A2_B1_B1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.45 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.27 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -1.2782028, 1.0692965, -0.4194033, 0.4558918, -1.7340946, 1.4886998
1: -0.8629500, 0.9084460, -0.2649211, 0.2740566, -1.1370065, 1.1733671
2: -0.9835748, 1.1688298, -0.2655452, 0.4088201, -1.3923949, 1.4343750
3: -1.1514845, 0.8033743, -0.2473766, 0.4075684, -1.5590529, 1.0507510
4: -1.1741228, 1.0863595, -0.3895604, 0.3490234, -1.5231462, 1.4759200
5: -0.8292906, 1.2996870, 0.3155125, 1.0969805, -1.9262710, 0.9841745
6: -0.9281736, 1.0723410, -0.3166411, 0.4244723, -1.3526459, 1.3889821
7: -0.9867958, 1.1181241, -0.3411733, 0.3191743, -1.3059701, 1.4592974
8: -1.2744329, 1.0770785, -0.4147728, 0.5091894, -1.7836223, 1.4918513
9: -1.0492498, 1.0596892, -0.3163287, 0.3995789, -1.4488287, 1.3760178

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of NS_B2_A2_A2_B1_B1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.38 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.33 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.7303391, 0.7696322, -0.5522239, 0.6160899, -1.3464290, 1.3218560
1: -0.5144022, 0.5342680, -0.3852801, 0.4014319, -0.9158340, 0.9195482
2: -0.5074266, 0.7994224, -0.3770197, 0.5938873, -1.1013138, 1.1764421
3: -0.5012291, 0.6053761, -0.3414238, 0.5108677, -1.0120969, 0.9467999
4: -0.6779469, 0.6634290, -0.5226315, 0.5082861, -1.1862330, 1.1860605
5: -0.1845481, 1.1549166, 0.0635770, 1.1232300, -1.3077781, 1.0913396
6: -0.5449648, 0.6826361, -0.4255004, 0.5531882, -1.0981530, 1.1081365
7: -0.5892066, 0.6865842, -0.4523025, 0.5036410, -1.0928476, 1.1388867
8: -0.7346758, 0.7937900, -0.5685194, 0.6517796, -1.3864553, 1.3623095
9: -0.6281897, 0.6950313, -0.4765466, 0.5414714, -1.1696610, 1.1715779

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of NS_B2_A2_A2_B1_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7994943, upper bound: 1.8015216
time: 1.49 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7994943, upper bound: 1.8015053
time: 1.66 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.0225120, 0.9226595, -0.5812300, 0.6494889, -1.6720009, 1.5038896
1: -0.6900922, 0.7231655, -0.4113484, 0.4267971, -1.1168892, 1.1345140
2: -0.7493664, 0.9958051, -0.3973115, 0.6389747, -1.3883412, 1.3931166
3: -0.8338242, 0.7078733, -0.3616793, 0.5312867, -1.3651109, 1.0695527
4: -0.9285325, 0.8845283, -0.5521665, 0.5365927, -1.4651251, 1.4366949
5: -0.5206762, 1.2211725, 0.0121073, 1.1281939, -1.6488700, 1.2090652
6: -0.7440170, 0.8729190, -0.4475703, 0.5787177, -1.3227347, 1.3204893
7: -0.7906154, 0.9112385, -0.4749938, 0.5409802, -1.3315957, 1.3862323
8: -1.0138474, 0.9388489, -0.5989981, 0.6826048, -1.6964521, 1.5378469
9: -0.8398656, 0.8864730, -0.5060623, 0.5722522, -1.4121177, 1.3925352

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_B2_A2_A2_B1_B2_A2_A1

### Relational analysis result of NS_B2_A2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8014497, upper bound: 1.8015053
time: 1.52 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8014497, upper bound: 1.8015053
time: 1.49 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.9317928, 0.8745389, -1.4563478, 1.1585714, -2.0903642, 2.3308868
1: -0.6322764, 0.6606359, -0.9808577, 1.0228373, -1.6551137, 1.6414937
2: -0.6702123, 0.9343356, -1.1369269, 1.2894864, -1.9596987, 2.0712624
3: -0.7239649, 0.6752693, -1.3654125, 0.8637218, -1.5876868, 2.0406818
4: -0.8447069, 0.8164514, -1.3331177, 1.2144201, -2.0591269, 2.1495690
5: -0.4133378, 1.1955799, -1.0372155, 1.3437510, -1.7570888, 2.2327952
6: -0.6815399, 0.8070214, -1.0431294, 1.2008965, -1.8824364, 1.8501508
7: -0.7259092, 0.8404481, -1.1088476, 1.2509400, -1.9768492, 1.9492958
8: -0.9230522, 0.8912461, -1.4489088, 1.1704319, -2.0934839, 2.3401549
9: -0.7676963, 0.8259603, -1.1883422, 1.1763136, -1.9440099, 2.0143025

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B2_A2_A2_B2_A1_A1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6974626, upper bound: 1.6596056
time: 1.51 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_A1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7081169, upper bound: 1.6596056
time: 1.38 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1.3143133, 1.0900248, -1.4777822, 1.1702266, -2.4845400, 2.5678072
1: -0.8873568, 0.9347174, -0.9952224, 1.0376430, -1.9249997, 1.9299399
2: -1.0167105, 1.1932213, -1.1560742, 1.3039964, -2.3207068, 2.3492956
3: -1.1963899, 0.8168917, -1.3916755, 0.8714217, -2.0678115, 2.2085671
4: -1.2089534, 1.1148518, -1.3531065, 1.2307181, -2.4396715, 2.4679585
5: -0.8729209, 1.3108215, -1.0627352, 1.3498073, -2.2227283, 2.3735566
6: -0.9543707, 1.1006138, -1.0579075, 1.2171001, -2.1714709, 2.1585212
7: -1.0145763, 1.1473186, -1.1245697, 1.2677228, -2.2822990, 2.2718883
8: -1.3112624, 1.0966337, -1.4704027, 1.1818736, -2.4931359, 2.5670364
9: -1.0788258, 1.0841987, -1.2055591, 1.1906333, -2.2694592, 2.2897577

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B2_A2_A2_B2_A2_A1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6974559, upper bound: 1.6596056
time: 1.61 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7080558, upper bound: 1.6596056
time: 1.49 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.09 seconds
NS_B1_A1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7663792, upper bound: 1.7355502
NS_B1_A1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7730046, upper bound: 1.7416129
NS_B1_A1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7416853, upper bound: 1.7139773
NS_B1_A1_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7355682, upper bound: 1.7139773
NS_B1_A1_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7475049, upper bound: 1.7139773
NS_B1_A1_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7397975, upper bound: 1.7139773
NS_B1_A1_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7322638, upper bound: 1.7139773
NS_B1_A1_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7139215, upper bound: 1.7139773
NS_B1_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7365488, upper bound: 1.7704756
NS_B1_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7417652, upper bound: 1.7763012
NS_B1_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7455782
NS_B1_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7387646
NS_B1_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7509678
NS_B1_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7425660
NS_B1_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7338122
NS_B1_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139773
NS_B1_A1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7365488, upper bound: 1.7703020
NS_B1_A1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7417652, upper bound: 1.7763001
NS_B1_A1_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7455782
NS_B1_A1_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7387646
NS_B1_A1_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7509678
NS_B1_A1_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7425548
NS_B1_A1_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7338122
NS_B1_A1_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7135653
NS_B1_A1_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7365488, upper bound: 1.7703020
NS_B1_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7417652, upper bound: 1.7763001
NS_B1_A1_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7455782
NS_B1_A1_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7387646
NS_B1_A1_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7509678
NS_B1_A1_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7425660
NS_B1_A1_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7338122
NS_B1_A1_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139773
NS_B1_A2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7541090, upper bound: 1.7139773
NS_B1_A2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7490314, upper bound: 1.7139773
NS_B1_A2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7416570, upper bound: 1.7139773
NS_B1_A2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7355521, upper bound: 1.7139773
NS_B1_A2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7474594, upper bound: 1.7139773
NS_B1_A2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7396925, upper bound: 1.7139773
NS_B1_A2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7322638, upper bound: 1.7139773
NS_B1_A2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7139215, upper bound: 1.7139773
NS_B1_A2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7541090, upper bound: 1.7139773
NS_B1_A2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7490314, upper bound: 1.7139773
NS_B1_A2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7416570, upper bound: 1.7139773
NS_B1_A2_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7355521, upper bound: 1.7139773
NS_B1_A2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7571323, upper bound: 1.7415417
NS_B1_A2_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7362933, upper bound: 1.7412851
NS_B1_A2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7650095, upper bound: 1.7416129
NS_B1_A2_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7417497, upper bound: 1.7416129
NS_B1_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7538090, upper bound: 1.7135653
NS_B1_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7485881, upper bound: 1.7135653
NS_B1_A2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7407451, upper bound: 1.7135653
NS_B1_A2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7350036, upper bound: 1.7135653
NS_B1_A2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7571030, upper bound: 1.7415417
NS_B1_A2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7362933, upper bound: 1.7412851
NS_B1_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7651389, upper bound: 1.7416129
NS_B1_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7417497, upper bound: 1.7416129
NS_B1_A2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7547507, upper bound: 1.7139773
NS_B1_A2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7492547, upper bound: 1.7139773
NS_B1_A2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7421488, upper bound: 1.7139773
NS_B1_A2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7357030, upper bound: 1.7139773
NS_B1_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7571895, upper bound: 1.7415417
NS_B1_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7362933, upper bound: 1.7412851
NS_B1_A2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7651663, upper bound: 1.7416129
NS_B1_A2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7417497, upper bound: 1.7416129
NS_B2_A1_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7814949
NS_B2_A1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7679741
NS_B2_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740677
NS_B2_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7513879
NS_B2_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7843712, upper bound: 1.7513879
NS_B2_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7703235, upper bound: 1.7513879
NS_B2_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7763739, upper bound: 1.7513879
NS_B2_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7513879
NS_B2_A1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7843712, upper bound: 1.7513879
NS_B2_A1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7702354, upper bound: 1.7513879
NS_B2_A1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7763739, upper bound: 1.7513879
NS_B2_A1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7513879
NS_B2_A1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7843712, upper bound: 1.7513879
NS_B2_A1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7702354, upper bound: 1.7513879
NS_B2_A1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7763739, upper bound: 1.7513879
NS_B2_A1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7513879
NS_B2_A1_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7813746
NS_B2_A1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7678785
NS_B2_A1_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7738816
NS_B2_A1_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_B2_A1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7813746
NS_B2_A1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7678785
NS_B2_A1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7903640, upper bound: 1.7861047
NS_B2_A1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
NS_B2_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7812601
NS_B2_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7675506
NS_B2_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7903640, upper bound: 1.7861047
NS_B2_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
NS_B2_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7814618
NS_B2_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7678959
NS_B2_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7903640, upper bound: 1.7861047
NS_B2_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7917661, upper bound: 1.7917661
NS_B2_A2_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7997147, upper bound: 1.8015053
NS_B2_A2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7996038, upper bound: 1.8015053
NS_B2_A2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8015216, upper bound: 1.8015053
NS_B2_A2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8015053, upper bound: 1.8015053
NS_B2_A2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8015215, upper bound: 1.7994943
NS_B2_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8015053, upper bound: 1.7994943
NS_B2_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8015053, upper bound: 1.8014497
NS_B2_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8015053, upper bound: 1.8014497
NS_B2_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_B2_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7994943, upper bound: 1.8015216
NS_B2_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7994943, upper bound: 1.8015053
NS_B2_A2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8014497, upper bound: 1.8015053
NS_B2_A2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.8014497, upper bound: 1.8015053
NS_B2_A2_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.6974626, upper bound: 1.6596056
NS_B2_A2_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7081169, upper bound: 1.6596056
NS_B2_A2_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.6974559, upper bound: 1.6596056
NS_B2_A2_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 5, lower bound: -1.7080558, upper bound: 1.6596056

## BFS NS instance: NS_B1_A1_B1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2722157, 0.2901469, -0.1702970, 0.1943630, -0.4665787, 0.4604439
1: -0.1574636, 0.1710072, -0.0888376, 0.0957307, -0.2531942, 0.2598448
2: -0.1576934, 0.2472006, -0.0900693, 0.1362191, -0.2939124, 0.3372698
3: -0.1528999, 0.2620480, -0.0894272, 0.1667855, -0.3196853, 0.3514752
4: -0.2442768, 0.2163621, -0.1530845, 0.1290651, -0.3733419, 0.3694467
5: 0.5512053, 1.0706398, 0.7065378, 1.0564712, -0.5052660, 0.3641021
6: -0.2051945, 0.2696620, -0.1313784, 0.1709222, -0.3761167, 0.4010403
7: -0.2300540, 0.1968392, -0.1512133, 0.1267633, -0.3568173, 0.3480526
8: -0.2510981, 0.3480920, -0.1423467, 0.2301941, -0.4812922, 0.4904387
9: -0.1880744, 0.2546818, -0.1098523, 0.1529034, -0.3409777, 0.3645340

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7364824, upper bound: 1.6909574
time: 1.52 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7315976, upper bound: 1.6909574
time: 1.59 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3199979, 0.3477426, -0.2410026, 0.2544099, -0.5744078, 0.5887452
1: -0.1928824, 0.2048221, -0.1352725, 0.1486072, -0.3414896, 0.3400946
2: -0.1925793, 0.3004013, -0.1353560, 0.2125949, -0.4051742, 0.4357573
3: -0.1853600, 0.3121713, -0.1316035, 0.2318267, -0.4171867, 0.4437748
4: -0.2897964, 0.2580681, -0.2157430, 0.1887318, -0.4785282, 0.4738111
5: 0.4720672, 1.0771158, 0.6023985, 1.0665071, -0.5944399, 0.4747173
6: -0.2419396, 0.3216739, -0.1820772, 0.2352042, -0.4771438, 0.5037512
7: -0.2678995, 0.2301269, -0.2060640, 0.1754321, -0.4433315, 0.4361908
8: -0.3058466, 0.4014271, -0.2156233, 0.3127572, -0.6186038, 0.6170503
9: -0.2254863, 0.3052084, -0.1632889, 0.2223450, -0.4478314, 0.4684972

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7452494, upper bound: 1.7024936
time: 1.53 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7401077, upper bound: 1.7024936
time: 1.63 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.5088056, 0.5705281, -0.2401253, 0.2534839, -0.7622894, 0.8106534
1: -0.3286074, 0.3351343, -0.1346902, 0.1479776, -0.4765849, 0.4698245
2: -0.3305404, 0.5192448, -0.1347532, 0.2116224, -0.5421629, 0.6539980
3: -0.3106038, 0.5087365, -0.1310049, 0.2310741, -0.5416779, 0.6397414
4: -0.4827939, 0.4171974, -0.2149907, 0.1879551, -0.6707490, 0.6321881
5: 0.1701163, 1.1201718, 0.6037972, 1.0663910, -0.8962747, 0.5163746
6: -0.3907738, 0.5201247, -0.1814538, 0.2342359, -0.6250097, 0.7015785
7: -0.4181771, 0.3731911, -0.2054248, 0.1748304, -0.5930075, 0.5786159
8: -0.5339028, 0.6049283, -0.2146482, 0.3117643, -0.8456671, 0.8195764
9: -0.3765771, 0.4991198, -0.1625924, 0.2214674, -0.5980445, 0.6617122

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B1_A1_A1_A2_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7179781, upper bound: 1.7024943
time: 1.56 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_A2_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7351849, upper bound: 1.7053725
time: 1.53 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.5217634, 0.5857811, -0.1970541, 0.2159280, -0.7376914, 0.7828352
1: -0.3378877, 0.3440498, -0.1060934, 0.1176702, -0.4555579, 0.4501432
2: -0.3400217, 0.5343949, -0.1078515, 0.1638699, -0.5038915, 0.6422465
3: -0.3191580, 0.5222430, -0.1044070, 0.1941157, -0.5132737, 0.6266500
4: -0.4962653, 0.4280660, -0.1788703, 0.1498282, -0.6460935, 0.6069363
5: 0.1494927, 1.1233683, 0.6684976, 1.0606881, -0.9111954, 0.4548707
6: -0.4010115, 0.5336790, -0.1508418, 0.1941617, -0.5951732, 0.6845208
7: -0.4285118, 0.3831685, -0.1740415, 0.1466049, -0.5751167, 0.5572101
8: -0.5497534, 0.6188277, -0.1703850, 0.2630056, -0.8127590, 0.7892127
9: -0.3870303, 0.5123639, -0.1308573, 0.1783853, -0.5654156, 0.6432211

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B1_A1_A1_A2_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7085998, upper bound: 1.7015961
time: 1.43 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_A2_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7293626, upper bound: 1.7053725
time: 1.45 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.4257017, 0.4620900, -0.2510940, 0.2650629, -0.6907645, 0.7131840
1: -0.2698935, 0.2790423, -0.1419726, 0.1558492, -0.4257427, 0.4210149
2: -0.2707633, 0.4151216, -0.1422893, 0.2237833, -0.4945467, 0.5574110
3: -0.2509044, 0.4128187, -0.1384888, 0.2404858, -0.4913901, 0.5513074
4: -0.3958783, 0.3565711, -0.2243987, 0.1976649, -0.5935432, 0.5809698
5: 0.3052413, 1.0983243, 0.5863073, 1.0678432, -0.7626020, 0.5120170
6: -0.3212095, 0.4310018, -0.1892495, 0.2463448, -0.5675542, 0.6202513
7: -0.3456738, 0.3270203, -0.2134170, 0.1823531, -0.5280269, 0.5404373
8: -0.4211602, 0.5162670, -0.2268404, 0.3241811, -0.7453413, 0.7431074
9: -0.3241383, 0.4046721, -0.1713021, 0.2324391, -0.5565774, 0.5759742

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A1_A2_A1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7293431, upper bound: 1.7020099
time: 1.75 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_A1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7385802, upper bound: 1.7024936
time: 1.52 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.4466513, 0.4840071, -0.2074712, 0.2229814, -0.6696326, 0.6914784
1: -0.2864332, 0.2983240, -0.1130098, 0.1248459, -0.4112791, 0.4113338
2: -0.2881615, 0.4360813, -0.1136676, 0.1754193, -0.4635808, 0.5497488
3: -0.2626386, 0.4303037, -0.1101244, 0.2030545, -0.4656931, 0.5404282
4: -0.4168931, 0.3817284, -0.1873953, 0.1590496, -0.5759428, 0.5691237
5: 0.2704636, 1.1027948, 0.6538697, 1.0620675, -0.7916039, 0.4489251
6: -0.3373573, 0.4527205, -0.1582456, 0.2019362, -0.5392935, 0.6109661
7: -0.3618791, 0.3531182, -0.1816320, 0.1530943, -0.5149735, 0.5347502
8: -0.4435487, 0.5398095, -0.1801637, 0.2747985, -0.7183472, 0.7199732
9: -0.3513710, 0.4216133, -0.1378999, 0.1888052, -0.5401762, 0.5595132

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A1_A2_A1_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7204025, upper bound: 1.7015593
time: 1.95 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_A1_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7307947, upper bound: 1.7024936
time: 1.48 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.7937604, 0.8471467, -0.2185919, 0.2307522, -1.0245126, 1.0657387
1: -0.5604739, 0.6177976, -0.1203933, 0.1325241, -0.6929979, 0.7381908
2: -0.5764267, 0.7833562, -0.1199582, 0.1877487, -0.7641754, 0.9033144
3: -0.4570611, 0.7200072, -0.1163129, 0.2125968, -0.6696579, 0.8363202
4: -0.7650811, 0.7985546, -0.1965209, 0.1688937, -0.9339747, 0.9950755
5: -0.3057563, 1.1768651, 0.6381329, 1.0635397, -1.3692961, 0.5387322
6: -0.6049057, 0.8125724, -0.1661493, 0.2104638, -0.8153695, 0.9787217
7: -0.6303794, 0.7855295, -0.1897348, 0.1600620, -0.7904414, 0.9752643
8: -0.8144982, 0.9298760, -0.1907131, 0.2873873, -1.1018854, 1.1205890
9: -0.8025853, 0.7023054, -0.1454933, 0.1999286, -1.0025139, 0.8477986

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A1_A2_A2_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7132470, upper bound: 1.7013255
time: 1.50 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_A2_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7227610, upper bound: 1.7024936
time: 1.52 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.8124149, 0.8666630, -0.1776581, 0.2012068, -1.0136218, 1.0443211
1: -0.5752016, 0.6349667, -0.0939184, 0.1026934, -0.6778951, 0.7288851
2: -0.5919189, 0.8020198, -0.0957126, 0.1426230, -0.7345420, 0.8977325
3: -0.4675099, 0.7355772, -0.0938882, 0.1754590, -0.6429689, 0.8294653
4: -0.7837939, 0.8209561, -0.1611308, 0.1354396, -0.9192334, 0.9820869
5: -0.3367241, 1.1808457, 0.6946995, 1.0578095, -1.3945336, 0.4861462
6: -0.6192847, 0.8319119, -0.1374635, 0.1780368, -0.7973216, 0.9693753
7: -0.6448095, 0.8087685, -0.1583680, 0.1330602, -0.7778697, 0.9671365
8: -0.8344342, 0.9508395, -0.1503322, 0.2405131, -1.0749474, 1.1011717
9: -0.8268347, 0.7173910, -0.1162594, 0.1608057, -0.9876404, 0.8336504

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A1_A2_A2_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6923329, upper bound: 1.7003516
time: 1.70 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_A2_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7023960, upper bound: 1.7024936
time: 1.59 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3161424, 0.3431695, -0.1952496, 0.2147063, -0.5308487, 0.5384191
1: -0.1900701, 0.2021369, -0.1048953, 0.1164272, -0.3064973, 0.3070322
2: -0.1898093, 0.2960881, -0.1068440, 0.1618694, -0.3516787, 0.4029322
3: -0.1827535, 0.3081916, -0.1034166, 0.1925674, -0.3753209, 0.4116082
4: -0.2861691, 0.2547565, -0.1773935, 0.1482308, -0.4343999, 0.4321499
5: 0.4783509, 1.0765796, 0.6710316, 1.0604494, -0.5820985, 0.4055480
6: -0.2389641, 0.3175441, -0.1495592, 0.1928149, -0.4317791, 0.4671033
7: -0.2648945, 0.2273940, -0.1727268, 0.1454807, -0.4103752, 0.4001207
8: -0.3014830, 0.3971925, -0.1686911, 0.2609627, -0.5624457, 0.5658836
9: -0.2225156, 0.3011729, -0.1296373, 0.1765803, -0.3990960, 0.4308102

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6927224, upper bound: 1.7419946
time: 1.42 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6927224, upper bound: 1.7373708
time: 1.38 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3957456, 0.4337395, -0.2433370, 0.2568743, -0.6526200, 0.6770766
1: -0.2464123, 0.2578302, -0.1368225, 0.1502826, -0.3966949, 0.3946527
2: -0.2464194, 0.3850225, -0.1369598, 0.2151833, -0.4616027, 0.5219823
3: -0.2340867, 0.3877597, -0.1331961, 0.2338299, -0.4679166, 0.5209558
4: -0.3656998, 0.3215644, -0.2177454, 0.1907982, -0.5564980, 0.5393099
5: 0.3533185, 1.0919043, 0.5986761, 1.0668161, -0.7134976, 0.4932281
6: -0.3003069, 0.4001048, -0.1837364, 0.2377813, -0.5380882, 0.5838412
7: -0.3254945, 0.2895421, -0.2077650, 0.1770331, -0.5025276, 0.4973071
8: -0.3917523, 0.4828788, -0.2182181, 0.3153999, -0.7071522, 0.7010969
9: -0.2884343, 0.3803437, -0.1651427, 0.2246799, -0.5131142, 0.5454864

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7503546
time: 1.46 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7454784
time: 1.34 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.3837475, 0.4220087, -0.4220042, 0.4732589, -0.8570064, 0.8440129
1: -0.2382428, 0.2492375, -0.2700693, 0.2785023, -0.5167452, 0.5193068
2: -0.2382973, 0.3724200, -0.2686050, 0.4132690, -0.6515663, 0.6410251
3: -0.2273095, 0.3772703, -0.2550983, 0.4214032, -0.6487128, 0.6323687
4: -0.3530644, 0.3113675, -0.3885458, 0.3489565, -0.7020209, 0.6999133
5: 0.3708934, 1.0892169, 0.2996046, 1.0904719, -0.7195784, 0.7896124
6: -0.2916573, 0.3883983, -0.3200198, 0.4350211, -0.7266784, 0.7084181
7: -0.3175461, 0.2767980, -0.3503746, 0.2995694, -0.6171155, 0.6271727
8: -0.3795615, 0.4703811, -0.4245900, 0.5176591, -0.8972206, 0.8949711
9: -0.2761880, 0.3701575, -0.3070168, 0.4145091, -0.6906971, 0.6771742

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B1_A2_B1_B2_B1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7026906, upper bound: 1.7223924
time: 1.47 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_B2_B1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7058700, upper bound: 1.7392093
time: 1.41 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.3357053, 0.3663746, -0.4429406, 0.4988531, -0.8345584, 0.8093152
1: -0.2043402, 0.2157626, -0.2858086, 0.2935275, -0.4978677, 0.5015713
2: -0.2038647, 0.3179729, -0.2841075, 0.4364811, -0.6403458, 0.6020805
3: -0.1959787, 0.3283861, -0.2693830, 0.4436771, -0.6396558, 0.5977691
4: -0.3045749, 0.2715600, -0.4087108, 0.3674897, -0.6720645, 0.6802707
5: 0.4464663, 1.0792999, 0.2644376, 1.0932437, -0.6467774, 0.8148624
6: -0.2540621, 0.3384994, -0.3360693, 0.4581341, -0.7121961, 0.6745687
7: -0.2801424, 0.2412608, -0.3671924, 0.3139284, -0.5940708, 0.6084532
8: -0.3236241, 0.4186811, -0.4488396, 0.5413599, -0.8649840, 0.8675207
9: -0.2375888, 0.3216490, -0.3236418, 0.4368485, -0.6744373, 0.6452909

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B1_A2_B1_B2_B2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7021173, upper bound: 1.7127946
time: 1.49 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_B2_B2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7058700, upper bound: 1.7328712
time: 1.29 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.4047267, 0.4422238, -0.3415296, 0.3732831, -0.7780099, 0.7837534
1: -0.2533341, 0.2640448, -0.2085886, 0.2198193, -0.4731534, 0.4726334
2: -0.2534106, 0.3941366, -0.2080493, 0.3244886, -0.5778992, 0.6021860
3: -0.2391561, 0.3953464, -0.1999160, 0.3343982, -0.5735542, 0.5952624
4: -0.3748380, 0.3314671, -0.3100547, 0.2765625, -0.6514005, 0.6415218
5: 0.3390812, 1.0938485, 0.4369739, 1.0801097, -0.7410285, 0.6568745
6: -0.3065627, 0.4092571, -0.2585568, 0.3447381, -0.6513008, 0.6678139
7: -0.3314212, 0.3008910, -0.2846819, 0.2453891, -0.5768102, 0.5855728
8: -0.4005690, 0.4926963, -0.3302159, 0.4250785, -0.8256475, 0.8229123
9: -0.2988782, 0.3877106, -0.2420764, 0.3277451, -0.6266233, 0.6297870

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A2_B2_B1_B1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7021370, upper bound: 1.7328366
time: 1.64 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_B1_B1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7420408
time: 1.35 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.3520201, 0.3858093, -0.3695360, 0.4065033, -0.7585235, 0.7553453
1: -0.2162410, 0.2271605, -0.2290177, 0.2393256, -0.4555666, 0.4561782
2: -0.2157188, 0.3362242, -0.2281709, 0.3558190, -0.5715379, 0.5643952
3: -0.2070079, 0.3452891, -0.2188489, 0.3633086, -0.5703164, 0.5641381
4: -0.3201128, 0.2855731, -0.3364042, 0.3006178, -0.6207306, 0.6219773
5: 0.4198762, 1.0818038, 0.3913283, 1.0840039, -0.6641278, 0.6904755
6: -0.2668043, 0.3559751, -0.2801707, 0.3747376, -0.6415420, 0.6361458
7: -0.2930188, 0.2528251, -0.3065105, 0.2652406, -0.5582594, 0.5593356
8: -0.3423253, 0.4366018, -0.3619131, 0.4558415, -0.7981669, 0.7985148
9: -0.2502227, 0.3387251, -0.2636551, 0.3570583, -0.6072810, 0.6023802

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A2_B2_B1_B2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7016862, upper bound: 1.7240526
time: 1.50 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_B1_B2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7336835
time: 1.31 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.3571479, 0.3919902, -0.6218109, 0.7057453, -1.0628932, 1.0138010
1: -0.2199812, 0.2307737, -0.4130385, 0.4150359, -0.6350172, 0.6438122
2: -0.2195610, 0.3419606, -0.4094230, 0.6380377, -0.8575987, 0.7513836
3: -0.2104744, 0.3506556, -0.3893934, 0.6237273, -0.8342018, 0.7400490
4: -0.3251615, 0.2899774, -0.5737565, 0.5173041, -0.8424655, 0.8637338
5: 0.4115188, 1.0827975, -0.0198379, 1.1190828, -0.7075641, 1.1026355
6: -0.2709420, 0.3614680, -0.4748644, 0.6449681, -0.9159101, 0.8363324
7: -0.2972068, 0.2564597, -0.5031390, 0.4440597, -0.7412665, 0.7595987
8: -0.3484108, 0.4422340, -0.6474352, 0.7329484, -1.0813591, 1.0896692
9: -0.2542489, 0.3440922, -0.4580310, 0.6211071, -0.8753560, 0.8021232

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A2_B2_B2_B1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7009683, upper bound: 1.7151101
time: 1.51 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_B2_B1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7245216
time: 1.26 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.3105115, 0.3364902, -0.6492878, 0.7383378, -1.0488492, 0.9857780
1: -0.1859626, 0.1982149, -0.4330810, 0.4341737, -0.6201363, 0.6312960
2: -0.1857636, 0.2897888, -0.4291644, 0.6687755, -0.8545392, 0.7189532
3: -0.1789470, 0.3023788, -0.4079684, 0.6520911, -0.8310382, 0.7103472
4: -0.2808714, 0.2499201, -0.5996078, 0.5409047, -0.8217760, 0.8495279
5: 0.4875284, 1.0757966, -0.0646207, 1.1229033, -0.6353750, 1.1404173
6: -0.2346185, 0.3115123, -0.4960697, 0.6744004, -0.9090190, 0.8075820
7: -0.2605056, 0.2234026, -0.5245551, 0.4635361, -0.7240417, 0.7479577
8: -0.2951100, 0.3910072, -0.6785328, 0.7631301, -1.0582402, 1.0695400
9: -0.2181771, 0.2952793, -0.4792019, 0.6498662, -0.8680434, 0.7744812

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B1_A2_B2_B2_B2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7000399, upper bound: 1.6909574
time: 1.38 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_B2_B2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7024936
time: 1.23 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3455543, 0.3780571, -0.2049562, 0.2212783, -0.5668325, 0.5830133
1: -0.2115243, 0.2226224, -0.1113399, 0.1231133, -0.3346376, 0.3339623
2: -0.2109409, 0.3289909, -0.1122632, 0.1726309, -0.3835718, 0.4412541
3: -0.2026368, 0.3385528, -0.1087440, 0.2008963, -0.4035331, 0.4472968
4: -0.3138411, 0.2800192, -0.1853369, 0.1568231, -0.4706642, 0.4653561
5: 0.4304144, 1.0806692, 0.6574016, 1.0617346, -0.6313201, 0.4232675
6: -0.2616629, 0.3490492, -0.1564580, 0.2000591, -0.4617219, 0.5055072
7: -0.2878186, 0.2482419, -0.1797993, 0.1515273, -0.4393460, 0.4280412
8: -0.3347710, 0.4294994, -0.1778027, 0.2719511, -0.6067221, 0.6073021
9: -0.2451773, 0.3319575, -0.1361994, 0.1862893, -0.4314666, 0.4681569

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B2_B1_B1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7157036, upper bound: 1.7547653
time: 1.44 seconds

## Relational analysis of NS_B1_A1_B2_B1_B1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7303349, upper bound: 1.7641877
time: 1.30 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4332341, 0.4699701, -0.2504181, 0.2643493, -0.6975834, 0.7203882
1: -0.2758404, 0.2859750, -0.1415239, 0.1553642, -0.4312046, 0.4274989
2: -0.2770190, 0.4226575, -0.1418251, 0.2230339, -0.5000529, 0.5644826
3: -0.2551235, 0.4191055, -0.1380276, 0.2399059, -0.4950294, 0.5571331
4: -0.4034342, 0.3656164, -0.2238190, 0.1970666, -0.6005008, 0.5894355
5: 0.2927372, 1.0999318, 0.5873852, 1.0677538, -0.7750165, 0.5125467
6: -0.3270155, 0.4388108, -0.1887692, 0.2455985, -0.5726140, 0.6275800
7: -0.3515005, 0.3364038, -0.2129245, 0.1818895, -0.5333900, 0.5493283
8: -0.4292100, 0.5247317, -0.2260890, 0.3234161, -0.7526261, 0.7508206
9: -0.3339299, 0.4107633, -0.1707656, 0.2317630, -0.5656928, 0.5815288

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B2_B1_B1_B1_A2_A1

### Relational analysis result of NS_B1_A1_B2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7273260, upper bound: 1.7461354
time: 1.48 seconds

## Relational analysis of NS_B1_A1_B2_B1_B1_B1_A2_A2

### Relational analysis result of NS_B1_A1_B2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7454784
time: 1.38 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.4195385, 0.4560174, -0.4138079, 0.4632391, -0.8827776, 0.8698253
1: -0.2650278, 0.2741486, -0.2639077, 0.2726201, -0.5376479, 0.5380563
2: -0.2656569, 0.4089555, -0.2625359, 0.4041821, -0.6698390, 0.6714914
3: -0.2474522, 0.4076807, -0.2495061, 0.4126835, -0.6601356, 0.6571867
4: -0.3896959, 0.3491852, -0.3806516, 0.3417013, -0.7313973, 0.7298368
5: 0.3152956, 1.0970091, 0.3133720, 1.0893867, -0.7740911, 0.7836371
6: -0.3167339, 0.4246123, -0.3137367, 0.4259727, -0.7427066, 0.7383490
7: -0.3412629, 0.3193425, -0.3437909, 0.2939481, -0.6352111, 0.6631334
8: -0.4149034, 0.5093413, -0.4150966, 0.5083807, -0.9232841, 0.9244379
9: -0.3164893, 0.3996880, -0.3005083, 0.4057634, -0.7222527, 0.7001963

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B2_B1_B1_B2_B1_A1

### Relational analysis result of NS_B1_A1_B2_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6852939, upper bound: 1.7275063
time: 1.36 seconds

## Relational analysis of NS_B1_A1_B2_B1_B1_B2_B1_A2

### Relational analysis result of NS_B1_A1_B2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7058700, upper bound: 1.7392093
time: 1.29 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.3634731, 0.3994548, -0.4373457, 0.4920131, -0.8554862, 0.8368005
1: -0.2245202, 0.2351369, -0.2816025, 0.2895121, -0.5140323, 0.5167394
2: -0.2242009, 0.3493219, -0.2799646, 0.4302777, -0.6544786, 0.6292865
3: -0.2146606, 0.3572515, -0.2655656, 0.4377246, -0.6523852, 0.6228170
4: -0.3317002, 0.2952961, -0.4033220, 0.3625369, -0.6942372, 0.6986181
5: 0.4014262, 1.0843221, 0.2738357, 1.0925031, -0.6910769, 0.8104864
6: -0.2759507, 0.3681012, -0.3317801, 0.4519571, -0.7279078, 0.6998813
7: -0.3022645, 0.2612887, -0.3626981, 0.3100910, -0.6123556, 0.6239868
8: -0.3561237, 0.4490360, -0.4423590, 0.5350261, -0.8911498, 0.8913950
9: -0.2593368, 0.3505736, -0.3191988, 0.4308787, -0.6902155, 0.6697724

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B2_B1_B1_B2_B2_A1

### Relational analysis result of NS_B1_A1_B2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6852939, upper bound: 1.7210920
time: 1.44 seconds

## Relational analysis of NS_B1_A1_B2_B1_B1_B2_B2_A2

### Relational analysis result of NS_B1_A1_B2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7058700, upper bound: 1.7328712
time: 1.30 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.4421011, 0.4792269, -0.3292448, 0.3587112, -0.8008123, 0.8084717
1: -0.2828260, 0.2941595, -0.1996275, 0.2112629, -0.4940889, 0.4937871
2: -0.2844065, 0.4315100, -0.1992230, 0.3107457, -0.5951522, 0.6307331
3: -0.2600942, 0.4264902, -0.1916112, 0.3217169, -0.5818111, 0.6181014
4: -0.4123097, 0.3762869, -0.2984967, 0.2660106, -0.6783203, 0.6747836
5: 0.2779821, 1.1018199, 0.4569961, 1.0784014, -0.8004193, 0.6448238
6: -0.3338481, 0.4479839, -0.2490761, 0.3315789, -0.6654270, 0.6970600
7: -0.3583447, 0.3474768, -0.2751069, 0.2366814, -0.5950260, 0.6225837
8: -0.4386658, 0.5346808, -0.3163121, 0.4115845, -0.8502502, 0.8509929
9: -0.3454316, 0.4179816, -0.2326110, 0.3148871, -0.6603187, 0.6505926

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B1_B2_B1_B1_A1

### Relational analysis result of NS_B1_A1_B2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6927224, upper bound: 1.7342452
time: 1.36 seconds

## Relational analysis of NS_B1_A1_B2_B1_B2_B1_B1_A2

### Relational analysis result of NS_B1_A1_B2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7420408
time: 1.51 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.3821244, 0.4203948, -0.3584784, 0.3933875, -0.7755119, 0.7788732
1: -0.2372608, 0.2480551, -0.2209519, 0.2316243, -0.4688851, 0.4690070
2: -0.2372593, 0.3706867, -0.2202266, 0.3434492, -0.5807086, 0.5909132
3: -0.2264044, 0.3758268, -0.2113740, 0.3518943, -0.5782987, 0.5872008
4: -0.3513261, 0.3102174, -0.3260010, 0.2911202, -0.6424463, 0.6362184
5: 0.3730924, 1.0888467, 0.4093500, 1.0824665, -0.7093741, 0.6794966
6: -0.2904671, 0.3868495, -0.2716372, 0.3628933, -0.6533604, 0.6584867
7: -0.3164524, 0.2755236, -0.2978922, 0.2574030, -0.5738554, 0.5734158
8: -0.3778843, 0.4687039, -0.3493985, 0.4436957, -0.8215801, 0.8181024
9: -0.2747367, 0.3687561, -0.2551354, 0.3454850, -0.6202217, 0.6238915

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B1_B2_B1_B2_A1

### Relational analysis result of NS_B1_A1_B2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6927224, upper bound: 1.7265538
time: 1.29 seconds

## Relational analysis of NS_B1_A1_B2_B1_B2_B1_B2_A2

### Relational analysis result of NS_B1_A1_B2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7336692
time: 1.33 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.3874792, 0.4257196, -0.5916956, 0.6700234, -1.0575026, 1.0174152
1: -0.2405005, 0.2519553, -0.3910708, 0.3940604, -0.6345609, 0.6430261
2: -0.2406832, 0.3764063, -0.3877861, 0.6043476, -0.8450308, 0.7641925
3: -0.2293906, 0.3805881, -0.3690346, 0.5926397, -0.8220303, 0.7496227
4: -0.3570612, 0.3140114, -0.5454224, 0.4914370, -0.8484982, 0.8594339
5: 0.3658372, 1.0900669, 0.0292452, 1.1148951, -0.7490579, 1.0608217
6: -0.2943932, 0.3919595, -0.4516227, 0.6127093, -0.9071025, 0.8435822
7: -0.3200603, 0.2797278, -0.4796664, 0.4227132, -0.7427735, 0.7593941
8: -0.3834177, 0.4742377, -0.6133507, 0.6998687, -1.0832863, 1.0875883
9: -0.2795240, 0.3733797, -0.4348273, 0.5895860, -0.8691100, 0.8082070

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B1_B2_B2_B1_B1

### Relational analysis result of NS_B1_A1_B2_B1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7006681, upper bound: 1.7151100
time: 1.87 seconds

## Relational analysis of NS_B1_A1_B2_B1_B2_B2_B1_B2

### Relational analysis result of NS_B1_A1_B2_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7245211
time: 1.50 seconds

## BFS NS instance: NS_B1_A1_B2_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.3379927, 0.3690875, -0.6183592, 0.7016511, -1.0396438, 0.9874467
1: -0.2060086, 0.2173560, -0.4105204, 0.4126319, -0.6186405, 0.6278764
2: -0.2055082, 0.3205317, -0.4069432, 0.6341760, -0.8396842, 0.7274748
3: -0.1975251, 0.3307473, -0.3870599, 0.6201641, -0.8176892, 0.7178073
4: -0.3067268, 0.2735245, -0.5705088, 0.5143394, -0.8210663, 0.8440333
5: 0.4427387, 1.0796177, -0.0142123, 1.1186029, -0.6758642, 1.0938301
6: -0.2558272, 0.3409495, -0.4722006, 0.6412706, -0.8970978, 0.8131500
7: -0.2819253, 0.2428821, -0.5004489, 0.4416130, -0.7235383, 0.7433310
8: -0.3262129, 0.4211935, -0.6435285, 0.7291570, -1.0553700, 1.0647221
9: -0.2393513, 0.3240431, -0.4553715, 0.6174945, -0.8568457, 0.7794147

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B1_B2_B2_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6995932, upper bound: 1.6907173
time: 1.30 seconds

## Relational analysis of NS_B1_A1_B2_B1_B2_B2_B2_B2

### Relational analysis result of NS_B1_A1_B2_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7019447
time: 1.35 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3662963, 0.4027783, -0.2474073, 0.2611712, -0.6274675, 0.6501856
1: -0.2265423, 0.2370794, -0.1395249, 0.1532037, -0.3797460, 0.3766043
2: -0.2262668, 0.3526230, -0.1397565, 0.2196959, -0.4459627, 0.4923795
3: -0.2165245, 0.3601942, -0.1359735, 0.2373224, -0.4538470, 0.4961677
4: -0.3346358, 0.2976644, -0.2212367, 0.1944015, -0.5290372, 0.5189011
5: 0.3969324, 1.0850186, 0.5921859, 1.0673550, -0.6704227, 0.4928328
6: -0.2781814, 0.3710546, -0.1866293, 0.2422750, -0.5204563, 0.5576839
7: -0.3045162, 0.2634628, -0.2107309, 0.1798247, -0.4843409, 0.4741936
8: -0.3595773, 0.4520648, -0.2227426, 0.3200077, -0.6795850, 0.6748074
9: -0.2616145, 0.3534595, -0.1683748, 0.2287515, -0.4903660, 0.5218343

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B2_B2_B1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7157036, upper bound: 1.7547653
time: 1.39 seconds

## Relational analysis of NS_B1_A1_B2_B2_B1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7303349, upper bound: 1.7641877
time: 1.40 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4608262, 0.4978940, -0.2940994, 0.3168990, -0.7777252, 0.7919934
1: -0.2969128, 0.3125023, -0.1739148, 0.1867119, -0.4836247, 0.4864171
2: -0.3010666, 0.4493615, -0.1738972, 0.2714627, -0.5725293, 0.6232587
3: -0.2707776, 0.4413823, -0.1678307, 0.2853293, -0.5561069, 0.6092130
4: -0.4302080, 0.3998168, -0.2653539, 0.2357337, -0.6659417, 0.6651707
5: 0.2452609, 1.1056274, 0.5144472, 1.0735371, -0.8282762, 0.5911802
6: -0.3482006, 0.4664819, -0.2219700, 0.2938206, -0.6420211, 0.6884519
7: -0.3721467, 0.3720715, -0.2476325, 0.2118477, -0.5839943, 0.6197040
8: -0.4577342, 0.5550127, -0.2764449, 0.3728652, -0.8305994, 0.8314576
9: -0.3686259, 0.4353714, -0.2054515, 0.2780318, -0.6466577, 0.6408229

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B2_B2_B1_B1_A2_A1

### Relational analysis result of NS_B1_A1_B2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7276628, upper bound: 1.7461354
time: 1.48 seconds

## Relational analysis of NS_B1_A1_B2_B2_B1_B1_A2_A2

### Relational analysis result of NS_B1_A1_B2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7454784
time: 1.51 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.4461418, 0.4832551, -0.4761862, 0.5330091, -0.9791509, 0.9594412
1: -0.2858657, 0.2981181, -0.3068131, 0.3136079, -0.5994735, 0.6049312
2: -0.2880015, 0.4353624, -0.3047960, 0.4751280, -0.7631295, 0.7401584
3: -0.2623996, 0.4297037, -0.2909474, 0.4734017, -0.7358013, 0.7206511
4: -0.4161720, 0.3813642, -0.4367458, 0.3922228, -0.8083948, 0.8181100
5: 0.2709213, 1.1026416, 0.2175065, 1.0988337, -0.8279123, 0.8851351
6: -0.3369455, 0.4519754, -0.3624782, 0.4889788, -0.8259243, 0.8144536
7: -0.3613231, 0.3527842, -0.3896361, 0.3408371, -0.7021602, 0.7424202
8: -0.4427806, 0.5390685, -0.4826187, 0.5729895, -1.0157701, 1.0216873
9: -0.3504369, 0.4217343, -0.3458282, 0.4686860, -0.8191229, 0.7675626

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B2_B2_B1_B2_B1_A1

### Relational analysis result of NS_B1_A1_B2_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6852939, upper bound: 1.7275063
time: 1.39 seconds

## Relational analysis of NS_B1_A1_B2_B2_B1_B2_B1_A2

### Relational analysis result of NS_B1_A1_B2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7058700, upper bound: 1.7392093
time: 1.29 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.3862157, 0.4244631, -0.4950467, 0.5553809, -0.9415966, 0.9195098
1: -0.2397360, 0.2510352, -0.3205707, 0.3267440, -0.5664800, 0.5716059
2: -0.2398753, 0.3750569, -0.3183466, 0.4962269, -0.7361021, 0.6934035
3: -0.2286860, 0.3794646, -0.3036974, 0.4928708, -0.7215568, 0.6831620
4: -0.3557079, 0.3131163, -0.4544905, 0.4084225, -0.7641304, 0.7676069
5: 0.3675491, 1.0897790, 0.1867667, 1.1014562, -0.7339071, 0.9030123
6: -0.2934669, 0.3907538, -0.3770338, 0.5091814, -0.8026483, 0.7677876
7: -0.3192091, 0.2787357, -0.4043361, 0.3542060, -0.6734151, 0.6830718
8: -0.3821120, 0.4729317, -0.5039647, 0.5937066, -0.9758186, 0.9768963
9: -0.2783945, 0.3722885, -0.3603601, 0.4884266, -0.7668211, 0.7326486

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B2_B2_B1_B2_B2_B1

### Relational analysis result of NS_B1_A1_B2_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7021268, upper bound: 1.7127946
time: 1.46 seconds

## Relational analysis of NS_B1_A1_B2_B2_B1_B2_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7058700, upper bound: 1.7328712
time: 1.33 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.4704055, 0.5074434, -0.3983883, 0.4362876, -0.9066931, 0.9058317
1: -0.3041193, 0.3218862, -0.2483765, 0.2596965, -0.5638158, 0.5702628
2: -0.3095897, 0.4584939, -0.2482900, 0.3877596, -0.6973493, 0.7067838
3: -0.2762431, 0.4490008, -0.2355949, 0.3900383, -0.6662815, 0.6845957
4: -0.4393648, 0.4118540, -0.3684444, 0.3241173, -0.7634821, 0.7802985
5: 0.2285217, 1.1075753, 0.3492087, 1.0924883, -0.8639665, 0.7583666
6: -0.3555430, 0.4759448, -0.3021857, 0.4027299, -0.7582728, 0.7781305
7: -0.3792077, 0.3846534, -0.3272208, 0.2929503, -0.6721581, 0.7118741
8: -0.4674891, 0.5654144, -0.3944000, 0.4856497, -0.9531388, 0.9598144
9: -0.3804917, 0.4442678, -0.2914068, 0.3825560, -0.7630478, 0.7356747

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B2_B2_B1_B1_A1

### Relational analysis result of NS_B1_A1_B2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6927224, upper bound: 1.7342452
time: 1.23 seconds

## Relational analysis of NS_B1_A1_B2_B2_B2_B1_B1_A2

### Relational analysis result of NS_B1_A1_B2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7420408
time: 1.43 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.4086581, 0.4458848, -0.4283330, 0.4651572, -0.8738153, 0.8742178
1: -0.2564378, 0.2667265, -0.2706325, 0.2808436, -0.5372814, 0.5373590
2: -0.2566609, 0.3980699, -0.2694857, 0.4187741, -0.6754351, 0.6675556
3: -0.2413580, 0.3986202, -0.2526857, 0.4158535, -0.6572115, 0.6513059
4: -0.3787816, 0.3361697, -0.3995403, 0.3530424, -0.7318240, 0.7357100
5: 0.3327681, 1.0946876, 0.3026431, 1.0991035, -0.7663354, 0.7920445
6: -0.3092623, 0.4133328, -0.3234730, 0.4324733, -0.7417356, 0.7368058
7: -0.3340333, 0.3057882, -0.3467819, 0.3315679, -0.6656013, 0.6525701
8: -0.4043738, 0.4971141, -0.4244013, 0.5170438, -0.9214176, 0.9215153
9: -0.3035523, 0.3908896, -0.3250854, 0.4076242, -0.7111765, 0.7159750

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B2_B2_B1_B2_A1

### Relational analysis result of NS_B1_A1_B2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6927224, upper bound: 1.7265538
time: 1.37 seconds

## Relational analysis of NS_B1_A1_B2_B2_B2_B1_B2_A2

### Relational analysis result of NS_B1_A1_B2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7336835
time: 1.35 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.4138273, 0.4506990, -0.7394567, 0.7651124, -1.1789397, 1.1901557
1: -0.2605188, 0.2702527, -0.5018703, 0.5005587, -0.7610776, 0.7721230
2: -0.2609349, 0.4032416, -0.4897073, 0.7410126, -1.0019474, 0.8929489
3: -0.2442534, 0.4029249, -0.4302573, 0.6840727, -0.9283260, 0.8331822
4: -0.3839668, 0.3423531, -0.7226260, 0.6535721, -1.0375389, 1.0649792
5: 0.3244668, 1.0957906, -0.1811716, 1.1678334, -0.8433666, 1.2769623
6: -0.3128120, 0.4186917, -0.5446470, 0.7415069, -1.0543189, 0.9633386
7: -0.3374681, 0.3122277, -0.5500199, 0.7328050, -1.0702730, 0.8622476
8: -0.4093763, 0.5029231, -0.7361143, 0.8432258, -1.2526021, 1.2390374
9: -0.3096987, 0.3950696, -0.6750057, 0.6680805, -0.9777792, 1.0700754

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B2_B2_B2_B1_B1

### Relational analysis result of NS_B1_A1_B2_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7009786, upper bound: 1.7151100
time: 1.42 seconds

## Relational analysis of NS_B1_A1_B2_B2_B2_B2_B1_B2

### Relational analysis result of NS_B1_A1_B2_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7245216
time: 1.43 seconds

## BFS NS instance: NS_B1_A1_B2_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.3590288, 0.3942235, -0.7690151, 0.7936099, -1.1526387, 1.1632386
1: -0.2213374, 0.2320789, -0.5238388, 0.5214328, -0.7427701, 0.7559177
2: -0.2209490, 0.3441259, -0.5106294, 0.7716267, -0.9925757, 0.8547553
3: -0.2117267, 0.3526190, -0.4471276, 0.7095546, -0.9212813, 0.7997466
4: -0.3270798, 0.2915688, -0.7533211, 0.6821238, -1.0092036, 1.0448899
5: 0.4084993, 1.0832258, -0.2271363, 1.1743633, -0.7658640, 1.3103621
6: -0.2724395, 0.3634526, -0.5656598, 0.7708666, -1.0433061, 0.9291123
7: -0.2987198, 0.2578668, -0.5693287, 0.7709242, -1.0696440, 0.8271955
8: -0.3506871, 0.4442691, -0.7657286, 0.8742148, -1.2249019, 1.2099977
9: -0.2557519, 0.3460310, -0.7082498, 0.6928251, -0.9485769, 1.0542808

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_B1_A1_B2_B2_B2_B2_B2_B1

### Relational analysis result of NS_B1_A1_B2_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7000965, upper bound: 1.6909574
time: 1.91 seconds

## Relational analysis of NS_B1_A1_B2_B2_B2_B2_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7028848, upper bound: 1.7024936
time: 1.20 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.3667461, 0.4033078, -0.2779954, 0.2972125, -0.6639585, 0.6813033
1: -0.2268645, 0.2373890, -0.1618085, 0.1751550, -0.4020195, 0.3991975
2: -0.2265960, 0.3531490, -0.1619730, 0.2536084, -0.4802044, 0.5151220
3: -0.2168215, 0.3606631, -0.1568432, 0.2681971, -0.4850186, 0.5175064
4: -0.3351033, 0.2980418, -0.2498435, 0.2214783, -0.5565816, 0.5478853
5: 0.3962164, 1.0851295, 0.5414970, 1.0714049, -0.6751885, 0.5436325
6: -0.2785368, 0.3715250, -0.2096251, 0.2760426, -0.5545794, 0.5811502
7: -0.3048750, 0.2638092, -0.2346967, 0.2008032, -0.5056782, 0.4985059
8: -0.3601276, 0.4525471, -0.2577926, 0.3546346, -0.7147622, 0.7103397
9: -0.2619775, 0.3539194, -0.1926639, 0.2608486, -0.5228261, 0.5465832

Time for backsubstitution: 0.93 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.90 + 596.65 = 600.55 seconds
