## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 21.8342855583


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167)
1: (-14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249)
2: (-17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199)
3: (-20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737)
4: (-18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998)
5: (-15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762)
6: (-14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840)
7: (-18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370)
8: (-19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403)
9: (-14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.96 + 5.94 = 7.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -21.8561417, upper bound: 21.8561417

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8493059, upper bound: 21.8496531
time: 32.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8540747, upper bound: 21.8540747
time: 4.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 37.11 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 37.11
Output dim: 7, lower bound: -21.8493059, upper bound: 21.8496531
NS_A2, status: Status.UNKNOWN, split count: 1, time: 37.11
Output dim: 7, lower bound: -21.8540747, upper bound: 21.8540747

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -13.8008862, 11.1241951, -15.3336906, 12.3649616, -26.1658478, 26.4578857
1: -11.3064289, 9.6706553, -12.6435766, 10.7632256, -22.0696526, 22.3142281
2: -13.9232807, 8.3834190, -15.5627575, 9.4551659, -23.3784447, 23.9461765
3: -16.6641884, 7.4775047, -18.5410156, 8.3144579, -24.9786453, 26.0185204
4: -15.0195122, 11.6458445, -16.6490269, 12.9874783, -28.0069904, 28.2948704
5: -12.8944426, 9.7707701, -14.3608150, 10.9263935, -23.8208351, 24.1315842
6: -11.8975430, 13.0537663, -13.2530251, 14.4575729, -26.3551159, 26.3067894
7: -14.9307690, 8.2593975, -16.5649471, 9.4673433, -24.3981133, 24.8243446
8: -15.2676067, 10.6232224, -17.0743561, 11.8228836, -27.0904846, 27.6975784
9: -11.8708916, 12.1745691, -13.3017979, 13.6113863, -25.4822769, 25.4763680

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8484479, upper bound: 21.8484479
time: 4.65 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8484479, upper bound: 21.8496531
time: 4.27 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -15.3707504, 12.3966303, -16.5219479, 13.3239517, -28.6947002, 28.9185791
1: -12.6879845, 10.7957373, -13.6715260, 11.5992622, -24.2872467, 24.4672623
2: -15.6171923, 9.5045547, -16.8527794, 10.2840271, -25.9012165, 26.3573341
3: -18.5917492, 8.3412170, -20.0064030, 8.9797173, -27.5714664, 28.3476200
4: -16.6844196, 13.0311213, -17.9138489, 14.0069723, -30.6913910, 30.9449692
5: -14.4047661, 10.9648666, -15.5046463, 11.8113689, -26.2161350, 26.4695129
6: -13.2970314, 14.4881277, -14.3123884, 15.5316496, -28.8286800, 28.8005161
7: -16.6001797, 9.5437651, -17.8029251, 10.4388733, -27.0390530, 27.3466892
8: -17.1392937, 11.8595333, -18.4767475, 12.7614994, -29.9007931, 30.3362770
9: -13.3502626, 13.6585722, -14.3928366, 14.7147503, -28.0650101, 28.0514088

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8496531, upper bound: 21.8493059
time: 4.03 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8496531, upper bound: 21.8540748
time: 8.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 14.14 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.14
Output dim: 7, lower bound: -21.8484479, upper bound: 21.8484479
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.14
Output dim: 7, lower bound: -21.8484479, upper bound: 21.8496531
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.14
Output dim: 7, lower bound: -21.8496531, upper bound: 21.8493059
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.14
Output dim: 7, lower bound: -21.8496531, upper bound: 21.8540748

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -13.8008862, 11.1241951, -13.8008862, 11.1241951, -24.9250793, 24.9250813
1: -11.3064289, 9.6706553, -11.3064289, 9.6706553, -20.9770851, 20.9770851
2: -13.9232807, 8.3834190, -13.9232807, 8.3834190, -22.3066978, 22.3066978
3: -16.6641884, 7.4775047, -16.6641884, 7.4775047, -24.1416931, 24.1416931
4: -15.0195122, 11.6458445, -15.0195122, 11.6458445, -26.6653519, 26.6653557
5: -12.8944426, 9.7707701, -12.8944426, 9.7707701, -22.6652126, 22.6652107
6: -11.8975430, 13.0537663, -11.8975430, 13.0537663, -24.9513092, 24.9513092
7: -14.9307690, 8.2593975, -14.9307690, 8.2593975, -23.1901665, 23.1901665
8: -15.2676067, 10.6232224, -15.2676067, 10.6232224, -25.8908272, 25.8908272
9: -11.8708916, 12.1745691, -11.8708916, 12.1745691, -24.0454597, 24.0454578

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8393301, upper bound: 21.8386781
time: 8.45 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
time: 5.26 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -13.8008862, 11.1241951, -15.3661613, 12.3923111, -26.1931953, 26.4903526
1: -11.3064289, 9.6706553, -12.6827984, 10.7920265, -22.0984535, 22.3534508
2: -13.9232807, 8.3834190, -15.6104975, 9.5011005, -23.4243813, 23.9939156
3: -16.6641884, 7.4775047, -18.5829277, 8.3370943, -25.0012798, 26.0604324
4: -15.0195122, 11.6458445, -16.6778984, 13.0272894, -28.0468025, 28.3237381
5: -12.8944426, 9.7707701, -14.4003925, 10.9606304, -23.8550720, 24.1711617
6: -11.8975430, 13.0537663, -13.2921429, 14.4835081, -26.3810463, 26.3459053
7: -14.9307690, 8.2593975, -16.5955353, 9.5389175, -24.4696846, 24.8549328
8: -15.2676067, 10.6232224, -17.1332855, 11.8550444, -27.1226501, 27.7565060
9: -11.8708916, 12.1745691, -13.3443470, 13.6532326, -25.5241203, 25.5189171

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 62

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8386781, upper bound: 21.8404069
time: 51.16 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8399317
time: 3.97 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -15.3707504, 12.3966303, -13.8008862, 11.1241951, -26.4949455, 26.1975174
1: -12.6879845, 10.7957373, -11.3064289, 9.6706553, -22.3586349, 22.1021652
2: -15.6171923, 9.5045547, -13.9232807, 8.3834190, -24.0006104, 23.4278336
3: -18.5917492, 8.3412170, -16.6641884, 7.4775047, -26.0692539, 25.0054054
4: -16.6844196, 13.0311213, -15.0195122, 11.6458445, -28.3302612, 28.0506325
5: -14.4047661, 10.9648666, -12.8944426, 9.7707701, -24.1755371, 23.8593063
6: -13.2970314, 14.4881277, -11.8975430, 13.0537663, -26.3507919, 26.3856697
7: -16.6001797, 9.5437651, -14.9307690, 8.2593975, -24.8595772, 24.4745331
8: -17.1392937, 11.8595333, -15.2676067, 10.6232224, -27.7625160, 27.1271400
9: -13.3502626, 13.6585722, -11.8708916, 12.1745691, -25.5248299, 25.5294647

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 62

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8404069, upper bound: 21.8393852
time: 5.45 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8399317, upper bound: 21.8390395
time: 9.51 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -15.3707504, 12.3966303, -15.3707504, 12.3966303, -27.7673798, 27.7673798
1: -12.6879845, 10.7957373, -12.6879845, 10.7957373, -23.4837227, 23.4837208
2: -15.6171923, 9.5045547, -15.6171923, 9.5045547, -25.1217422, 25.1217461
3: -18.5917492, 8.3412170, -18.5917492, 8.3412170, -26.9329643, 26.9329643
4: -16.6844196, 13.0311213, -16.6844196, 13.0311213, -29.7155418, 29.7155399
5: -14.4047661, 10.9648666, -14.4047661, 10.9648666, -25.3696327, 25.3696327
6: -13.2970314, 14.4881277, -13.2970314, 14.4881277, -27.7851601, 27.7851582
7: -16.6001797, 9.5437651, -16.6001797, 9.5437651, -26.1439438, 26.1439438
8: -17.1392937, 11.8595333, -17.1392937, 11.8595333, -28.9988270, 28.9988270
9: -13.3502626, 13.6585722, -13.3502626, 13.6585722, -27.0088348, 27.0088348

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8426305, upper bound: 21.8498452
time: 5.86 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8399317, upper bound: 21.8390395
time: 15.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 23.52 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -21.8393301, upper bound: 21.8386781
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -21.8386781, upper bound: 21.8404069
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8399317
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -21.8404069, upper bound: 21.8393852
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -21.8399317, upper bound: 21.8390395
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -21.8426305, upper bound: 21.8498452
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -21.8399317, upper bound: 21.8390395

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -13.5293608, 10.9044218, -13.8008862, 11.1241951, -24.6535568, 24.7053070
1: -11.0691395, 9.4746132, -11.3064289, 9.6706553, -20.7397919, 20.7810402
2: -13.6389389, 8.1812315, -13.9232807, 8.3834190, -22.0223579, 22.1045113
3: -16.3401451, 7.3277054, -16.6641884, 7.4775047, -23.8176498, 23.9918938
4: -14.7288342, 11.4129982, -15.0195122, 11.6458445, -26.3746719, 26.4325104
5: -12.6351261, 9.5637131, -12.8944426, 9.7707701, -22.4058952, 22.4581528
6: -11.6568165, 12.8070030, -11.8975430, 13.0537663, -24.7105827, 24.7045460
7: -14.6401396, 8.0313616, -14.9307690, 8.2593975, -22.8995361, 22.9621277
8: -14.9482651, 10.4154682, -15.2676067, 10.6232224, -25.5714874, 25.6830730
9: -11.6089573, 11.9218502, -11.8708916, 12.1745691, -23.7835236, 23.7927418

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 62

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
time: 3.63 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
time: 3.25 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -15.0915852, 12.1424866, -13.4359407, 10.8298330, -25.9214172, 25.5784264
1: -12.3697929, 10.5368824, -10.9890270, 9.4086514, -21.7784424, 21.5259094
2: -15.2441425, 8.9238262, -13.5419865, 8.1144981, -23.3586388, 22.4658108
3: -18.3550167, 8.1237764, -16.2285957, 7.2769852, -25.6319981, 24.3523712
4: -16.4554272, 12.7176752, -14.6286087, 11.3347845, -27.7902107, 27.3462830
5: -14.1039047, 10.5977879, -12.5469456, 9.4938469, -23.5977516, 23.1447334
6: -12.9762039, 14.2952442, -11.5756130, 12.7221012, -25.6983051, 25.8708553
7: -16.3145561, 8.7697496, -14.5401535, 7.9589558, -24.2735119, 23.3099022
8: -16.6293354, 11.5954409, -14.8403416, 10.3446217, -26.9739571, 26.4357834
9: -12.8913727, 13.2788534, -11.5225725, 11.8367348, -24.7281075, 24.8014259

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
time: 8.58 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
time: 12.28 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -13.8008862, 11.1241951, -14.9850283, 12.0846615, -25.8855476, 26.1092224
1: -11.3064289, 9.6706553, -12.3518419, 10.5236378, -21.8300667, 22.0224953
2: -13.9232807, 8.3834190, -15.2012596, 9.2177925, -23.1410713, 23.5846786
3: -16.6641884, 7.4775047, -18.1310253, 8.1251526, -24.7893410, 25.6085300
4: -15.0195122, 11.6458445, -16.2741032, 12.6945667, -27.7140789, 27.9199448
5: -12.8944426, 9.7707701, -14.0355606, 10.6674213, -23.5618629, 23.8063316
6: -11.8975430, 13.0537663, -12.9500742, 14.1432076, -26.0407505, 26.0038414
7: -14.9307690, 8.2593975, -16.1970062, 9.2072430, -24.1380119, 24.4564037
8: -15.2676067, 10.6232224, -16.6730213, 11.5532055, -26.8208103, 27.2962437
9: -11.8708916, 12.1745691, -12.9854832, 13.2984982, -25.1693897, 25.1600494

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 62

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8365330, upper bound: 21.8375066
time: 6.51 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375501, upper bound: 21.8385511
time: 5.69 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -13.4359407, 10.8298330, -16.3695488, 13.1842480, -26.6201897, 27.1993828
1: -10.9890270, 9.4086514, -13.4992399, 11.4613771, -22.4504051, 22.9078865
2: -13.5419865, 8.1144981, -16.6124249, 9.8443432, -23.3863297, 24.7269230
3: -16.2285957, 7.2769852, -19.9207230, 8.8242130, -25.0528088, 27.1977081
4: -14.6286087, 11.3347845, -17.8196239, 13.8355217, -28.4641304, 29.1544075
5: -12.5469456, 9.4938469, -15.3310604, 11.5695200, -24.1164627, 24.8249054
6: -11.5756130, 12.7221012, -14.1070232, 15.4775095, -27.0531158, 26.8291245
7: -14.5401535, 7.9589558, -17.6993828, 9.8006115, -24.3407631, 25.6583385
8: -14.8403416, 10.3446217, -18.1398811, 12.5880108, -27.4283524, 28.4844990
9: -11.5225725, 11.8367348, -14.0993576, 14.4870129, -26.0095825, 25.9360924

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 254

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8359393, upper bound: 21.8367839
time: 5.40 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8372094, upper bound: 21.8380608
time: 21.36 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -14.9850283, 12.0846615, -13.8008862, 11.1241951, -26.1092224, 25.8855476
1: -12.3518419, 10.5236378, -11.3064289, 9.6706553, -22.0224953, 21.8300667
2: -15.2012596, 9.2177925, -13.9232807, 8.3834190, -23.5846786, 23.1410713
3: -18.1310253, 8.1251526, -16.6641884, 7.4775047, -25.6085300, 24.7893410
4: -16.2741032, 12.6945667, -15.0195122, 11.6458445, -27.9199448, 27.7140770
5: -14.0355606, 10.6674213, -12.8944426, 9.7707701, -23.8063316, 23.5618629
6: -12.9500742, 14.1432076, -11.8975430, 13.0537663, -26.0038376, 26.0407505
7: -16.1970062, 9.2072430, -14.9307690, 8.2593975, -24.4564037, 24.1380081
8: -16.6730213, 11.5532055, -15.2676067, 10.6232224, -27.2962437, 26.8208122
9: -12.9854832, 13.2984982, -11.8708916, 12.1745691, -25.1600456, 25.1693897

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 62

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375066, upper bound: 21.8365330
time: 5.92 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8385511, upper bound: 21.8375501
time: 7.16 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -16.3695488, 13.1842480, -13.4359407, 10.8298330, -27.1993828, 26.6201897
1: -13.4992399, 11.4613771, -10.9890270, 9.4086514, -22.9078865, 22.4504051
2: -16.6124249, 9.8443432, -13.5419865, 8.1144981, -24.7269230, 23.3863297
3: -19.9207230, 8.8242130, -16.2285957, 7.2769852, -27.1977062, 25.0528088
4: -17.8196239, 13.8355217, -14.6286087, 11.3347845, -29.1544075, 28.4641304
5: -15.3310604, 11.5695200, -12.5469456, 9.4938469, -24.8249073, 24.1164627
6: -14.1070232, 15.4775095, -11.5756130, 12.7221012, -26.8291245, 27.0531178
7: -17.6993828, 9.8006115, -14.5401535, 7.9589558, -25.6583385, 24.3407650
8: -18.1398811, 12.5880108, -14.8403416, 10.3446217, -28.4844971, 27.4283524
9: -14.0993576, 14.4870129, -11.5225725, 11.8367348, -25.9360924, 26.0095825

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 254

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8367839, upper bound: 21.8359393
time: 14.38 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8380608, upper bound: 21.8372094
time: 4.36 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -15.3707504, 12.3966303, -14.9850283, 12.0846615, -27.4554119, 27.3816586
1: -12.6879845, 10.7957373, -12.3518419, 10.5236378, -23.2116203, 23.1475792
2: -15.6171923, 9.5045547, -15.2012596, 9.2177925, -24.8349838, 24.7058144
3: -18.5917492, 8.3412170, -18.1310253, 8.1251526, -26.7168980, 26.4722424
4: -16.6844196, 13.0311213, -16.2741032, 12.6945667, -29.3789787, 29.3052254
5: -14.4047661, 10.9648666, -14.0355606, 10.6674213, -25.0721874, 25.0004272
6: -13.2970314, 14.4881277, -12.9500742, 14.1432076, -27.4402390, 27.4382019
7: -16.6001797, 9.5437651, -16.1970062, 9.2072430, -25.8074207, 25.7407722
8: -17.1392937, 11.8595333, -16.6730213, 11.5532055, -28.6924992, 28.5325546
9: -13.3502626, 13.6585722, -12.9854832, 13.2984982, -26.6487579, 26.6440544

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8489286, upper bound: 21.8489336
time: 3.86 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8489286, upper bound: 21.8489336
time: 5.92 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -14.8981209, 12.0149288, -16.3695488, 13.1842480, -28.0823689, 28.3844738
1: -12.2787876, 10.4638100, -13.4992399, 11.4613771, -23.7401638, 23.9630470
2: -15.1106987, 9.1619778, -16.6124249, 9.8443432, -24.9550419, 25.7744026
3: -18.0266399, 8.0786352, -19.9207230, 8.8242130, -26.8508530, 27.9993591
4: -16.1817913, 12.6213331, -17.8196239, 13.8355217, -30.0173130, 30.4409561
5: -13.9541426, 10.6037693, -15.3310604, 11.5695200, -25.5236626, 25.9348278
6: -12.8748760, 14.0644703, -14.1070232, 15.4775095, -28.3523865, 28.1714935
7: -16.1050720, 9.1451435, -17.6993828, 9.8006115, -25.9056816, 26.8445263
8: -16.5737419, 11.4864922, -18.1398811, 12.5880108, -29.1617527, 29.6263714
9: -12.9080181, 13.2200222, -14.0993576, 14.4870129, -27.3950272, 27.3193798

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 62

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8466222, upper bound: 21.8466254
time: 4.93 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8475192, upper bound: 21.8475193
time: 3.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 10.90 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8375492, upper bound: 21.8375492
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8365330, upper bound: 21.8375066
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8375501, upper bound: 21.8385511
NS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8359393, upper bound: 21.8367839
NS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8372094, upper bound: 21.8380608
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8375066, upper bound: 21.8365330
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8385511, upper bound: 21.8375501
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8367839, upper bound: 21.8359393
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8380608, upper bound: 21.8372094
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8489286, upper bound: 21.8489336
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8489286, upper bound: 21.8489336
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8466222, upper bound: 21.8466254
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 10.90
Output dim: 7, lower bound: -21.8475192, upper bound: 21.8475193

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13.5293608, 10.9044218, -13.5293608, 10.9044218, -24.4337807, 24.4337826
1: -11.0691395, 9.4746132, -11.0691395, 9.4746132, -20.5437508, 20.5437489
2: -13.6389389, 8.1812315, -13.6389389, 8.1812315, -21.8201714, 21.8201714
3: -16.3401451, 7.3277054, -16.3401451, 7.3277054, -23.6678486, 23.6678505
4: -14.7288342, 11.4129982, -14.7288342, 11.4129982, -26.1418285, 26.1418285
5: -12.6351261, 9.5637131, -12.6351261, 9.5637131, -22.1988373, 22.1988373
6: -11.6568165, 12.8070030, -11.6568165, 12.8070030, -24.4638176, 24.4638176
7: -14.6401396, 8.0313616, -14.6401396, 8.0313616, -22.6715012, 22.6715012
8: -14.9482651, 10.4154682, -14.9482651, 10.4154682, -25.3637333, 25.3637314
9: -11.6089573, 11.9218502, -11.6089573, 11.9218502, -23.5308075, 23.5308075

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8374195, upper bound: 21.8368898
time: 5.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8393301, upper bound: 21.8386781
time: 6.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -13.5293608, 10.9044218, -15.0915852, 12.1424866, -25.6718483, 25.9960060
1: -11.0691395, 9.4746132, -12.3697929, 10.5368824, -21.6060219, 21.8444061
2: -13.6389389, 8.1812315, -15.2441425, 8.9238262, -22.5627632, 23.4253731
3: -16.3401451, 7.3277054, -18.3550167, 8.1237764, -24.4639187, 25.6827183
4: -14.7288342, 11.4129982, -16.4554272, 12.7176752, -27.4465065, 27.8684216
5: -12.6351261, 9.5637131, -14.1039047, 10.5977879, -23.2329082, 23.6676178
6: -11.6568165, 12.8070030, -12.9762039, 14.2952442, -25.9520588, 25.7832050
7: -14.6401396, 8.0313616, -16.3145561, 8.7697496, -23.4098892, 24.3459167
8: -14.9482651, 10.4154682, -16.6293354, 11.5954409, -26.5437050, 27.0448036
9: -11.6089573, 11.9218502, -12.8913727, 13.2788534, -24.8878098, 24.8132229

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8363803, upper bound: 21.8356398
time: 6.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8359990, upper bound: 21.8354325
time: 8.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.0915852, 12.1424866, -13.5277195, 10.9028854, -25.9944687, 25.6702061
1: -12.3697929, 10.5368824, -11.0674953, 9.4731989, -21.8429909, 21.6043777
2: -15.2441425, 8.9238262, -13.6370430, 8.1802559, -23.4243965, 22.5608673
3: -18.3550167, 8.1237764, -16.3376045, 7.3269258, -25.6819363, 24.4613800
4: -16.4554272, 12.7176752, -14.7275791, 11.4110813, -27.8665085, 27.4452553
5: -14.1039047, 10.5977879, -12.6331024, 9.5624866, -23.6663914, 23.2308884
6: -12.9762039, 14.2952442, -11.6552782, 12.8056231, -25.7818260, 25.9505234
7: -16.3145561, 8.7697496, -14.6387291, 8.0298748, -24.3444309, 23.4084778
8: -16.6293354, 11.5954409, -14.9462013, 10.4141560, -27.0434914, 26.5416393
9: -12.8913727, 13.2788534, -11.6072845, 11.9203300, -24.8117027, 24.8861389

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8344893, upper bound: 21.8346591
time: 9.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8343881, upper bound: 21.8343881
time: 7.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.0915852, 12.1424866, -15.0915852, 12.1424866, -27.2340717, 27.2340717
1: -12.3697929, 10.5368824, -12.3697929, 10.5368824, -22.9066753, 22.9066753
2: -15.2441425, 8.9238262, -15.2441425, 8.9238262, -24.1679649, 24.1679649
3: -18.3550167, 8.1237764, -18.3550167, 8.1237764, -26.4787884, 26.4787884
4: -16.4554272, 12.7176752, -16.4554272, 12.7176752, -29.1730995, 29.1731014
5: -14.1039047, 10.5977879, -14.1039047, 10.5977879, -24.7016926, 24.7016926
6: -12.9762039, 14.2952442, -12.9762039, 14.2952442, -27.2714481, 27.2714481
7: -16.3145561, 8.7697496, -16.3145561, 8.7697496, -25.0843048, 25.0843048
8: -16.6293354, 11.5954409, -16.6293354, 11.5954409, -28.2247772, 28.2247772
9: -12.8913727, 13.2788534, -12.8913727, 13.2788534, -26.1702271, 26.1702271

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8346591, upper bound: 21.8344893
time: 3.27 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8343881, upper bound: 21.8343881
time: 5.86 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -13.3618679, 10.7716866, -13.3218327, 10.7540846, -24.1159515, 24.0935192
1: -10.9222641, 9.3527517, -10.9094257, 9.3439102, -20.2661705, 20.2621765
2: -13.4616814, 8.0816183, -13.4474773, 8.1255474, -21.5872288, 21.5290909
3: -16.1231499, 7.2378817, -16.0713654, 7.2268286, -23.3499775, 23.3092461
4: -14.5417366, 11.2657909, -14.4781876, 11.2564278, -25.7981625, 25.7439785
5: -12.4733849, 9.4455423, -12.4432192, 9.4522820, -21.9256630, 21.8887615
6: -11.5085735, 12.6460285, -11.4860592, 12.6042480, -24.1128197, 24.1320858
7: -14.4534836, 7.9293270, -14.4092131, 8.0122261, -22.4657097, 22.3385391
8: -14.7586727, 10.2847662, -14.7502327, 10.2695513, -25.0282211, 25.0349998
9: -11.4590330, 11.7699347, -11.4705410, 11.7731562, -23.2321892, 23.2404747

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 62

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8365330, upper bound: 21.8375066
time: 18.38 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8365330, upper bound: 21.8375066
time: 6.19 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -13.5763054, 10.9424839, -14.2714005, 11.5153790, -25.0916843, 25.2138844
1: -11.1098518, 9.5066414, -11.7291012, 10.0129519, -21.1228027, 21.2357388
2: -13.6848373, 8.2234230, -14.4369907, 8.7096491, -22.3944855, 22.6604137
3: -16.3875370, 7.3534737, -17.2546310, 7.7325573, -24.1200943, 24.6081047
4: -14.7747831, 11.4495544, -15.5117817, 12.0654688, -26.8402519, 26.9613342
5: -12.6783752, 9.6021700, -13.3482418, 10.1307888, -22.8091640, 22.9504070
6: -11.6966991, 12.8451900, -12.3109322, 13.4921360, -25.1888332, 25.1561222
7: -14.6855907, 8.0812950, -15.4319706, 8.6337557, -23.3193474, 23.5132656
8: -15.0041294, 10.4482050, -15.8233995, 10.9938469, -25.9979763, 26.2716045
9: -11.6540718, 11.9645071, -12.3109808, 12.6301966, -24.2842674, 24.2754860

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375501, upper bound: 21.8385511
time: 8.75 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375501, upper bound: 21.8385511
time: 5.45 seconds

## BFS NS instance: NS_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -13.0206356, 10.5002251, -14.7265511, 11.8612881, -24.8819160, 25.2267761
1: -10.6284790, 9.1119270, -12.0749655, 10.2951603, -20.9236355, 21.1868858
2: -13.1097603, 7.8346596, -14.8900242, 8.7821665, -21.8919258, 22.7246838
3: -15.7238674, 7.0522823, -17.8828735, 7.9424825, -23.6663494, 24.9351540
4: -14.1755724, 10.9828806, -16.0390701, 12.4195175, -26.5950890, 27.0219460
5: -12.1549768, 9.1890202, -13.7626657, 10.3737679, -22.5287437, 22.9516869
6: -11.2137508, 12.3356819, -12.6697731, 13.9460163, -25.1597672, 25.0054550
7: -14.0881958, 7.6607347, -15.9201393, 8.6603289, -22.7485237, 23.5808735
8: -14.3644810, 10.0285263, -16.2596016, 11.3257694, -25.6902485, 26.2881279
9: -11.1448812, 11.4595232, -12.6177483, 12.9890709, -24.1339531, 24.0772705

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8352058, upper bound: 21.8360525
time: 9.29 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8352058, upper bound: 21.8367839
time: 34.99 seconds

## BFS NS instance: NS_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -13.2299089, 10.6658230, -15.7274818, 12.6632767, -25.8931847, 26.3933048
1: -10.8107252, 9.2609692, -12.9369488, 10.9971800, -21.8079052, 22.1979179
2: -13.3261433, 7.9715624, -15.9279261, 9.3920212, -22.7181644, 23.8994865
3: -15.9786406, 7.1648502, -19.1263695, 8.4721842, -24.4508247, 26.2912178
4: -14.4040203, 11.1601524, -17.1266518, 13.2710438, -27.6750603, 28.2868042
5: -12.3526630, 9.3409462, -14.7120380, 11.0874777, -23.4401379, 24.0529842
6: -11.3954067, 12.5306263, -13.5349951, 14.8820648, -26.2774715, 26.0656204
7: -14.3152943, 7.8057995, -16.9999199, 9.3018818, -23.6171761, 24.8057194
8: -14.6023750, 10.1867676, -17.3845406, 12.0880585, -26.6904316, 27.5713081
9: -11.3318729, 11.6482677, -13.4950962, 13.8856583, -25.2175312, 25.1433601

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8353526, upper bound: 21.8361624
time: 4.40 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8353526, upper bound: 21.8380608
time: 5.52 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -13.3218327, 10.7540846, -13.3618679, 10.7716866, -24.0935192, 24.1159515
1: -10.9094257, 9.3439102, -10.9222641, 9.3527517, -20.2621765, 20.2661705
2: -13.4474773, 8.1255474, -13.4616814, 8.0816183, -21.5290909, 21.5872288
3: -16.0713654, 7.2268286, -16.1231499, 7.2378817, -23.3092461, 23.3499794
4: -14.4781876, 11.2564278, -14.5417366, 11.2657909, -25.7439785, 25.7981606
5: -12.4432192, 9.4522820, -12.4733849, 9.4455423, -21.8887615, 21.9256630
6: -11.4860592, 12.6042480, -11.5085735, 12.6460285, -24.1320877, 24.1128216
7: -14.4092131, 8.0122261, -14.4534836, 7.9293270, -22.3385391, 22.4657097
8: -14.7502327, 10.2695513, -14.7586727, 10.2847662, -25.0349960, 25.0282230
9: -11.4705410, 11.7731562, -11.4590330, 11.7699347, -23.2404747, 23.2321892

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 62

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375066, upper bound: 21.8365330
time: 7.64 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8375066, upper bound: 21.8365330
time: 5.94 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -14.2714005, 11.5153790, -13.5763054, 10.9424839, -25.2138844, 25.0916843
1: -11.7291012, 10.0129519, -11.1098518, 9.5066414, -21.2357407, 21.1228027
2: -14.4369907, 8.7096491, -13.6848373, 8.2234230, -22.6604137, 22.3944855
3: -17.2546310, 7.7325573, -16.3875370, 7.3534737, -24.6081047, 24.1200924
4: -15.5117817, 12.0654688, -14.7747831, 11.4495544, -26.9613323, 26.8402519
5: -13.3482418, 10.1307888, -12.6783752, 9.6021700, -22.9504128, 22.8091640
6: -12.3109322, 13.4921360, -11.6966991, 12.8451900, -25.1561184, 25.1888294
7: -15.4319706, 8.6337557, -14.6855907, 8.0812950, -23.5132656, 23.3193474
8: -15.8233995, 10.9938469, -15.0041294, 10.4482050, -26.2716045, 25.9979763
9: -12.3109808, 12.6301966, -11.6540718, 11.9645071, -24.2754841, 24.2842674

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 254

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8385511, upper bound: 21.8375501
time: 17.50 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8385511, upper bound: 21.8375501
time: 7.99 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -14.7265511, 11.8612881, -13.0206356, 10.5002251, -25.2267761, 24.8819237
1: -12.0749655, 10.2951603, -10.6284790, 9.1119270, -21.1868877, 20.9236336
2: -14.8900242, 8.7821665, -13.1097603, 7.8346596, -22.7246838, 21.8919258
3: -17.8828735, 7.9424825, -15.7238674, 7.0522823, -24.9351540, 23.6663494
4: -16.0390701, 12.4195175, -14.1755724, 10.9828806, -27.0219460, 26.5950890
5: -13.7626657, 10.3737679, -12.1549768, 9.1890202, -22.9516830, 22.5287437
6: -12.6697731, 13.9460163, -11.2137508, 12.3356819, -25.0054550, 25.1597652
7: -15.9201393, 8.6603289, -14.0881958, 7.6607347, -23.5808735, 22.7485237
8: -16.2596016, 11.3257694, -14.3644810, 10.0285263, -26.2881279, 25.6902504
9: -12.6177483, 12.9890709, -11.1448812, 11.4595232, -24.0772705, 24.1339531

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8360525, upper bound: 21.8352058
time: 8.09 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8360525, upper bound: 21.8359393
time: 23.66 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -15.7274818, 12.6632767, -13.2299089, 10.6658230, -26.3933048, 25.8931847
1: -12.9369488, 10.9971800, -10.8107252, 9.2609692, -22.1979179, 21.8079052
2: -15.9279261, 9.3920212, -13.3261433, 7.9715624, -23.8994865, 22.7181644
3: -19.1263695, 8.4721842, -15.9786406, 7.1648502, -26.2912178, 24.4508247
4: -17.1266518, 13.2710438, -14.4040203, 11.1601524, -28.2868042, 27.6750641
5: -14.7120380, 11.0874777, -12.3526630, 9.3409462, -24.0529823, 23.4401398
6: -13.5349951, 14.8820648, -11.3954067, 12.5306263, -26.0656204, 26.2774696
7: -16.9999199, 9.3018818, -14.3152943, 7.8057995, -24.8057194, 23.6171741
8: -17.3845406, 12.0880585, -14.6023750, 10.1867676, -27.5713081, 26.6904335
9: -13.4950962, 13.8856583, -11.3318729, 11.6482677, -25.1433601, 25.2175312

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8361624, upper bound: 21.8353526
time: 6.84 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8361624, upper bound: 21.8372094
time: 6.91 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -14.9850283, 12.0846615, -14.9850283, 12.0846615, -27.0696907, 27.0696907
1: -12.3518419, 10.5236378, -12.3518419, 10.5236378, -22.8754787, 22.8754787
2: -15.2012596, 9.2177925, -15.2012596, 9.2177925, -24.4190521, 24.4190521
3: -18.1310253, 8.1251526, -18.1310253, 8.1251526, -26.2561779, 26.2561779
4: -16.2741032, 12.6945667, -16.2741032, 12.6945667, -28.9686661, 28.9686661
5: -14.0355606, 10.6674213, -14.0355606, 10.6674213, -24.7029819, 24.7029819
6: -12.9500742, 14.1432076, -12.9500742, 14.1432076, -27.0932808, 27.0932808
7: -16.1970062, 9.2072430, -16.1970062, 9.2072430, -25.4042454, 25.4042473
8: -16.6730213, 11.5532055, -16.6730213, 11.5532055, -28.2262249, 28.2262268
9: -12.9854832, 13.2984982, -12.9854832, 13.2984982, -26.2839813, 26.2839813

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8473536, upper bound: 21.8476817
time: 4.12 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8480215, upper bound: 21.8483197
time: 14.87 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 7.91 + 609.45 = 617.35 seconds
