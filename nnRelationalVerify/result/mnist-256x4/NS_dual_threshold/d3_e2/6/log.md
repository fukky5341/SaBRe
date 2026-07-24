## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.783554394


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955)
1: (-0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281)
2: (-0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888)
3: (-0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397)
4: (-0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903)
5: (-0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065)
6: (-0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174)
7: (-0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887)
8: (-0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897)
9: (-0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.48 + 2.96 = 5.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7995453

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7979994
time: 1.83 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
time: 1.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.64 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.64
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7979994
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.64
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.2186534, 1.0904168, 0.1728942, 1.0965585, -0.8779051, 0.9175225
1: -0.3835300, 0.3779540, -0.4121212, 0.4078297, -0.7913597, 0.7900753
2: -0.2930107, 0.4877503, -0.3169437, 0.5166870, -0.8096977, 0.8046940
3: -0.2813488, 0.3669437, -0.2990719, 0.3966398, -0.6779886, 0.6660156
4: -0.4004387, 0.3811752, -0.4305013, 0.4228392, -0.8232780, 0.8116764
5: -0.4304177, 0.5497525, -0.4623537, 0.5819111, -1.0123287, 1.0121062
6: -0.3121916, 0.4397539, -0.3371477, 0.4755749, -0.7877666, 0.7769016
7: -0.4026238, 0.4266526, -0.4283904, 0.4551131, -0.8577368, 0.8550431
8: -0.4137142, 0.5166412, -0.4508296, 0.5498505, -0.9635647, 0.9674708
9: -0.3955224, 0.4956400, -0.4246632, 0.5256995, -0.9212219, 0.9203031

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
time: 1.52 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
time: 1.59 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0554657, 1.1135265, 0.1780109, 1.0959148, -1.0404491, 0.9355155
1: -0.5040710, 0.4977783, -0.4085843, 0.4043854, -0.9084563, 0.9063627
2: -0.3895540, 0.5955865, -0.3141212, 0.5136554, -0.9032094, 0.9097077
3: -0.3476263, 0.5043057, -0.2971106, 0.3925033, -0.7401296, 0.8014163
4: -0.5146475, 0.5465473, -0.4272584, 0.4181393, -0.9327868, 0.9738057
5: -0.5675435, 0.6660394, -0.4581623, 0.5785463, -1.1460898, 1.1242018
6: -0.4086787, 0.5632119, -0.3344338, 0.4715997, -0.8802783, 0.8976456
7: -0.4989775, 0.5481607, -0.4256903, 0.4515102, -0.9504876, 0.9738510
8: -0.5505396, 0.6379345, -0.4469061, 0.5463300, -1.0968696, 1.0848405
9: -0.5090230, 0.6205251, -0.4214398, 0.5220659, -1.0310888, 1.0419650

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7979261
time: 1.58 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7978884
time: 1.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.63 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.63
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.63
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.63
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7979261
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.63
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7978884

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.2186534, 1.0904168, 0.2186534, 1.0904168, -0.8717634, 0.8717634
1: -0.3835300, 0.3779540, -0.3835300, 0.3779540, -0.7614840, 0.7614840
2: -0.2930107, 0.4877503, -0.2930107, 0.4877503, -0.7807610, 0.7807610
3: -0.2813488, 0.3669437, -0.2813488, 0.3669437, -0.6482925, 0.6482925
4: -0.4004387, 0.3811752, -0.4004387, 0.3811752, -0.7816138, 0.7816138
5: -0.4304177, 0.5497525, -0.4304177, 0.5497525, -0.9801701, 0.9801701
6: -0.3121916, 0.4397539, -0.3121916, 0.4397539, -0.7519455, 0.7519455
7: -0.4026238, 0.4266526, -0.4026238, 0.4266526, -0.8292764, 0.8292764
8: -0.4137142, 0.5166412, -0.4137142, 0.5166412, -0.9303554, 0.9303554
9: -0.3955224, 0.4956400, -0.3955224, 0.4956400, -0.8911623, 0.8911623

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7994740, upper bound: 0.7978884
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7994303, upper bound: 0.7978884
time: 1.60 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.2186534, 1.0904168, 0.0554657, 1.1135265, -0.8948731, 1.0349511
1: -0.3835300, 0.3779540, -0.5040710, 0.4977783, -0.8813083, 0.8820250
2: -0.2930107, 0.4877503, -0.3895540, 0.5955865, -0.8885972, 0.8773042
3: -0.2813488, 0.3669437, -0.3476263, 0.5043057, -0.7856544, 0.7145700
4: -0.4004387, 0.3811752, -0.5146475, 0.5465473, -0.9469860, 0.8958226
5: -0.4304177, 0.5497525, -0.5675435, 0.6660394, -1.0964570, 1.1172960
6: -0.3121916, 0.4397539, -0.4086787, 0.5632119, -0.8754035, 0.8484325
7: -0.4026238, 0.4266526, -0.4989775, 0.5481607, -0.9507844, 0.9256301
8: -0.4137142, 0.5166412, -0.5505396, 0.6379345, -1.0516487, 1.0671808
9: -0.3955224, 0.4956400, -0.5090230, 0.6205251, -1.0160475, 1.0046629

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7994740, upper bound: 0.7978884
time: 1.48 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7994303, upper bound: 0.7978884
time: 1.57 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0713612, 1.1125772, 0.2928108, 1.0818772, -1.0105159, 0.8197664
1: -0.4982231, 0.4923072, -0.3488300, 0.3403404, -0.8385634, 0.8411372
2: -0.3845747, 0.5906250, -0.2582583, 0.4473447, -0.8319193, 0.8488833
3: -0.3434428, 0.4975443, -0.2536862, 0.3368065, -0.6802493, 0.7512305
4: -0.5092120, 0.5393289, -0.3601892, 0.3296285, -0.8388405, 0.8995181
5: -0.5592690, 0.6590858, -0.3873346, 0.5028519, -1.0621209, 1.0464203
6: -0.4046773, 0.5490538, -0.2809067, 0.3789232, -0.7836006, 0.8299605
7: -0.4947097, 0.5419717, -0.3695946, 0.3872306, -0.8819404, 0.9115663
8: -0.5436518, 0.6305671, -0.3640777, 0.4676886, -1.0113404, 0.9946448
9: -0.5039980, 0.6147144, -0.3560219, 0.4538925, -0.9578905, 0.9707363

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7979261
time: 1.53 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7979261
time: 1.46 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0685196, 1.1126741, 0.1662178, 1.1008993, -1.0323797, 0.9464563
1: -0.4989128, 0.4929237, -0.4338769, 0.4297293, -0.9286422, 0.9268006
2: -0.3852040, 0.5912036, -0.3331917, 0.5354938, -0.9206978, 0.9243953
3: -0.3440599, 0.4983316, -0.3075773, 0.4223219, -0.7663819, 0.8059089
4: -0.5098630, 0.5401040, -0.4501979, 0.4536780, -0.9635410, 0.9903020
5: -0.5604199, 0.6600924, -0.4827965, 0.5976924, -1.1581123, 1.1428890
6: -0.4050857, 0.5517133, -0.3554471, 0.4750994, -0.8801850, 0.9071604
7: -0.4951873, 0.5427332, -0.4456475, 0.4764174, -0.9716048, 0.9883807
8: -0.5445178, 0.6316414, -0.4722920, 0.5662023, -1.1107202, 1.1039335
9: -0.5045514, 0.6153742, -0.4454677, 0.5486810, -1.0532323, 1.0608418

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7978884
time: 1.53 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7978884
time: 1.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.72 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 0, lower bound: -0.7994740, upper bound: 0.7978884
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 0, lower bound: -0.7994303, upper bound: 0.7978884
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 0, lower bound: -0.7994740, upper bound: 0.7978884
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 0, lower bound: -0.7994303, upper bound: 0.7978884
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7979261
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7979261
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7978884
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 0, lower bound: -0.7978884, upper bound: 0.7978884

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.3246102, 1.0769563, 0.2339103, 1.0894706, -0.7648604, 0.8430460
1: -0.3294400, 0.3204069, -0.3790835, 0.3735105, -0.7029505, 0.6994904
2: -0.2393385, 0.4249448, -0.2885624, 0.4830324, -0.7223709, 0.7135072
3: -0.2396817, 0.3205543, -0.2771504, 0.3630683, -0.6027500, 0.5977046
4: -0.3388830, 0.3058940, -0.3955525, 0.3744636, -0.7133466, 0.7014465
5: -0.3658661, 0.4785203, -0.4235291, 0.5429722, -0.9088383, 0.9020494
6: -0.2650444, 0.3553236, -0.3083974, 0.4261406, -0.6911850, 0.6637209
7: -0.3520835, 0.3657104, -0.3986058, 0.4217098, -0.7737933, 0.7643163
8: -0.3408940, 0.4424144, -0.4066183, 0.5093018, -0.8501959, 0.8490327
9: -0.3346326, 0.4303319, -0.3909624, 0.4906474, -0.8252800, 0.8212943

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7927590, upper bound: 0.7932710
time: 1.86 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7927590, upper bound: 0.7926701
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.2042515, 1.0957124, 0.2311157, 1.0895779, -0.8853264, 0.8645967
1: -0.4030026, 0.4022098, -0.3796666, 0.3740703, -0.7770728, 0.7818764
2: -0.3108344, 0.5113100, -0.2891755, 0.4836363, -0.7944707, 0.8004856
3: -0.2924436, 0.3842381, -0.2778123, 0.3635962, -0.6560398, 0.6620504
4: -0.4239259, 0.4158337, -0.3961943, 0.3752792, -0.7992051, 0.8120279
5: -0.4483839, 0.5716736, -0.4246128, 0.5440047, -0.9923885, 0.9962864
6: -0.3335912, 0.4461096, -0.3088305, 0.4287143, -0.7623055, 0.7549401
7: -0.4240327, 0.4467190, -0.3991017, 0.4223891, -0.8464218, 0.8458207
8: -0.4410098, 0.5389712, -0.4076349, 0.5104128, -0.9514226, 0.9466060
9: -0.4188015, 0.5196810, -0.3915209, 0.4912729, -0.9100744, 0.9112018

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7932710, upper bound: 0.7926701
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7926701, upper bound: 0.7926701
time: 1.72 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.3246102, 1.0769563, 0.0713612, 1.1125772, -0.7879670, 1.0055950
1: -0.3294400, 0.3204069, -0.4982231, 0.4923072, -0.8217472, 0.8186300
2: -0.2393385, 0.4249448, -0.3845747, 0.5906250, -0.8299636, 0.8095194
3: -0.2396817, 0.3205543, -0.3434428, 0.4975443, -0.7372260, 0.6639971
4: -0.3388830, 0.3058940, -0.5092120, 0.5393289, -0.8782119, 0.8151059
5: -0.3658661, 0.4785203, -0.5592690, 0.6590858, -1.0249518, 1.0377893
6: -0.2650444, 0.3553236, -0.4046773, 0.5490538, -0.8140982, 0.7600008
7: -0.3520835, 0.3657104, -0.4947097, 0.5419717, -0.8940552, 0.8604202
8: -0.3408940, 0.4424144, -0.5436518, 0.6305671, -0.9714612, 0.9860661
9: -0.3346326, 0.4303319, -0.5039980, 0.6147144, -0.9493470, 0.9343300

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7927590, upper bound: 0.7916380
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7927590, upper bound: 0.7912376
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.2042515, 1.0957124, 0.0685196, 1.1126741, -0.9084226, 1.0271928
1: -0.4030026, 0.4022098, -0.4989128, 0.4929237, -0.8959262, 0.9011226
2: -0.3108344, 0.5113100, -0.3852040, 0.5912036, -0.9020380, 0.8965141
3: -0.2924436, 0.3842381, -0.3440599, 0.4983316, -0.7907752, 0.7282981
4: -0.4239259, 0.4158337, -0.5098630, 0.5401040, -0.9640299, 0.9256967
5: -0.4483839, 0.5716736, -0.5604199, 0.6600924, -1.1084763, 1.1320935
6: -0.3335912, 0.4461096, -0.4050857, 0.5517133, -0.8853046, 0.8511953
7: -0.4240327, 0.4467190, -0.4951873, 0.5427332, -0.9667659, 0.9419063
8: -0.4410098, 0.5389712, -0.5445178, 0.6316414, -1.0726511, 1.0834889
9: -0.4188015, 0.5196810, -0.5045514, 0.6153742, -1.0341758, 1.0242324

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7932710, upper bound: 0.7912376
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7926701, upper bound: 0.7912376
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0713612, 1.1125772, 0.3246102, 1.0769563, -1.0055950, 0.7879670
1: -0.4982231, 0.4923072, -0.3294400, 0.3204069, -0.8186300, 0.8217472
2: -0.3845747, 0.5906250, -0.2393385, 0.4249448, -0.8095194, 0.8299636
3: -0.3434428, 0.4975443, -0.2396817, 0.3205543, -0.6639971, 0.7372260
4: -0.5092120, 0.5393289, -0.3388830, 0.3058940, -0.8151059, 0.8782119
5: -0.5592690, 0.6590858, -0.3658661, 0.4785203, -1.0377893, 1.0249518
6: -0.4046773, 0.5490538, -0.2650444, 0.3553236, -0.7600008, 0.8140982
7: -0.4947097, 0.5419717, -0.3520835, 0.3657104, -0.8604202, 0.8940552
8: -0.5436518, 0.6305671, -0.3408940, 0.4424144, -0.9860661, 0.9714612
9: -0.5039980, 0.6147144, -0.3346326, 0.4303319, -0.9343300, 0.9493470

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7916380, upper bound: 0.7913886
time: 1.53 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7913886
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0713612, 1.1125772, 0.1840029, 1.0987369, -1.0273757, 0.9285743
1: -0.4982231, 0.4923072, -0.4143402, 0.4108763, -0.9090994, 0.9066474
2: -0.3845747, 0.5906250, -0.3233736, 0.5251237, -0.9096984, 0.9139987
3: -0.3434428, 0.4975443, -0.3020474, 0.3918061, -0.7352489, 0.7995917
4: -0.5092120, 0.5393289, -0.4369877, 0.4105251, -0.9197371, 0.9763166
5: -0.5592690, 0.6590858, -0.4615013, 0.5855693, -1.1448383, 1.1205871
6: -0.4046773, 0.5490538, -0.3348240, 0.4627579, -0.8674352, 0.8838778
7: -0.4947097, 0.5419717, -0.4292265, 0.4614596, -0.9561694, 0.9711981
8: -0.5436518, 0.6305671, -0.4465668, 0.5543190, -1.0979707, 1.0771339
9: -0.5039980, 0.6147144, -0.4302959, 0.5343053, -1.0383034, 1.0450102

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7916380, upper bound: 0.7913886
time: 2.04 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7913886
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0685196, 1.1126741, 0.2032693, 1.0957124, -1.0271928, 0.9094048
1: -0.4989128, 0.4929237, -0.4056863, 0.4022098, -0.9011226, 0.8986099
2: -0.3852040, 0.5912036, -0.3109086, 0.5113100, -0.8965141, 0.9021121
3: -0.3440599, 0.4983316, -0.2924436, 0.3893203, -0.7333802, 0.7907752
4: -0.5098630, 0.5401040, -0.4243894, 0.4158337, -0.9256967, 0.9644935
5: -0.5604199, 0.6600924, -0.4501432, 0.5716736, -1.1320935, 1.1102357
6: -0.4050857, 0.5517133, -0.3335912, 0.4472847, -0.8523704, 0.8853046
7: -0.4951873, 0.5427332, -0.4240327, 0.4478862, -0.9430735, 0.9667659
8: -0.5445178, 0.6316414, -0.4412549, 0.5389712, -1.0834889, 1.0728962
9: -0.5045514, 0.6153742, -0.4196412, 0.5196810, -1.0242324, 1.0350155

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7916380
time: 1.46 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7912376
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0685196, 1.1126741, 0.0406959, 1.1184639, -1.0499443, 1.0719782
1: -0.4989128, 0.4929237, -0.5293424, 0.5230279, -1.0219407, 1.0222660
2: -0.3852040, 0.5912036, -0.4087535, 0.6173909, -1.0025949, 0.9999571
3: -0.3440599, 0.4983316, -0.3588644, 0.5340789, -0.8781388, 0.8571960
4: -0.5098630, 0.5401040, -0.5375974, 0.5819517, -1.0918148, 1.0777014
5: -0.5604199, 0.6600924, -0.5935465, 0.6858038, -1.2462237, 1.2536390
6: -0.4050857, 0.5517133, -0.4294931, 0.5692922, -0.9743779, 0.9812064
7: -0.4951873, 0.5427332, -0.5188447, 0.5731676, -1.0683548, 1.0615779
8: -0.5445178, 0.6316414, -0.5773978, 0.6584346, -1.2029524, 1.2090392
9: -0.5045514, 0.6153742, -0.5329273, 0.6470520, -1.1516035, 1.1483015

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7916380
time: 1.71 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7912376
time: 1.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.02 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7927590, upper bound: 0.7932710
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7927590, upper bound: 0.7926701
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7932710, upper bound: 0.7926701
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7926701, upper bound: 0.7926701
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7927590, upper bound: 0.7916380
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7927590, upper bound: 0.7912376
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7932710, upper bound: 0.7912376
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7926701, upper bound: 0.7912376
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7916380, upper bound: 0.7913886
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7913886
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7916380, upper bound: 0.7913886
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7913886
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7916380
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7912376
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7916380
NS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.02
Output dim: 0, lower bound: -0.7912376, upper bound: 0.7912376

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.3246102, 1.0769563, 0.3810282, 1.0689805, -0.7443702, 0.6959281
1: -0.3294400, 0.3204069, -0.2962954, 0.2889529, -0.6183928, 0.6167023
2: -0.2393385, 0.4249448, -0.2102395, 0.3875544, -0.6268929, 0.6351843
3: -0.2396817, 0.3205543, -0.2157398, 0.2908480, -0.5305297, 0.5362941
4: -0.3388830, 0.3058940, -0.3084385, 0.2713432, -0.6102262, 0.6143324
5: -0.3658661, 0.4785203, -0.3318299, 0.4342905, -0.8001565, 0.8103502
6: -0.2650444, 0.3553236, -0.2362154, 0.3211720, -0.5862164, 0.5915389
7: -0.3520835, 0.3657104, -0.3210292, 0.3294711, -0.6815547, 0.6867396
8: -0.3408940, 0.4424144, -0.3082706, 0.4001795, -0.7410735, 0.7506850
9: -0.3346326, 0.4303319, -0.3011312, 0.3876245, -0.7222571, 0.7314631

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7925177, upper bound: 0.7924309
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7925177, upper bound: 0.7924309
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.3530945, 1.0728551, 0.3432153, 1.0742253, -0.7211308, 0.7296398
1: -0.3125902, 0.3045264, -0.3183533, 0.3100343, -0.6226245, 0.6228797
2: -0.2246469, 0.4060288, -0.2297424, 0.4125623, -0.6372092, 0.6357713
3: -0.2275940, 0.3054843, -0.2317863, 0.3106607, -0.5382547, 0.5372707
4: -0.3235123, 0.2883705, -0.3288432, 0.2943927, -0.6179049, 0.6172137
5: -0.3486819, 0.4561352, -0.3546419, 0.4638610, -0.8125430, 0.8107771
6: -0.2504518, 0.3380812, -0.2554868, 0.3440613, -0.5945131, 0.5935680
7: -0.3364048, 0.3472523, -0.3418426, 0.3535409, -0.6899457, 0.6890949
8: -0.3242080, 0.4210909, -0.3298445, 0.4284866, -0.7526946, 0.7509354
9: -0.3176938, 0.4087698, -0.3235514, 0.4162482, -0.7339420, 0.7323212

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7874587, upper bound: 0.7838832
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7839765, upper bound: 0.7838832
time: 1.87 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.3450854, 1.0739660, 0.2311157, 1.0895779, -0.7444925, 0.8428503
1: -0.3172624, 0.3089916, -0.3796666, 0.3740703, -0.6913326, 0.6886582
2: -0.2287777, 0.4113258, -0.2891755, 0.4836363, -0.7124140, 0.7005013
3: -0.2309927, 0.3096809, -0.2778123, 0.3635962, -0.5945889, 0.5874932
4: -0.3278340, 0.2932526, -0.3961943, 0.3752792, -0.7031132, 0.6894469
5: -0.3535138, 0.4623984, -0.4246128, 0.5440047, -0.8975185, 0.8870111
6: -0.2545337, 0.3429292, -0.3088305, 0.4287143, -0.6832480, 0.6517596
7: -0.3408131, 0.3523504, -0.3991017, 0.4223891, -0.7632022, 0.7514521
8: -0.3287777, 0.4270866, -0.4076349, 0.5104128, -0.8391905, 0.8347214
9: -0.3224426, 0.4148325, -0.3915209, 0.4912729, -0.8137155, 0.8063533

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7924309, upper bound: 0.7924309
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7924309, upper bound: 0.7926701
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.3120683, 1.0788932, 0.2675073, 1.0845070, -0.7724387, 0.8113859
1: -0.3372357, 0.3278563, -0.3596870, 0.3517414, -0.6889771, 0.6875433
2: -0.2467339, 0.4335788, -0.2692478, 0.4599729, -0.7067068, 0.7028265
3: -0.2451270, 0.3270723, -0.2626983, 0.3460548, -0.5911818, 0.5897707
4: -0.3465969, 0.3153112, -0.3728287, 0.3426687, -0.6892656, 0.6881399
5: -0.3742081, 0.4882120, -0.4013359, 0.5177941, -0.8920022, 0.8895479
6: -0.2713640, 0.3640856, -0.2893170, 0.3999988, -0.6713628, 0.6534026
7: -0.3590405, 0.3740931, -0.3791853, 0.3997012, -0.7587417, 0.7532785
8: -0.3494780, 0.4523564, -0.3786553, 0.4834147, -0.8328927, 0.8310118
9: -0.3428765, 0.4396604, -0.3678850, 0.4668938, -0.8097703, 0.8075454

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7838832, upper bound: 0.7872906
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7838832, upper bound: 0.7838832
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.3246102, 1.0769563, 0.2425796, 1.0893581, -0.7647479, 0.8343767
1: -0.3294400, 0.3204069, -0.3780392, 0.3726310, -0.7020710, 0.6984461
2: -0.2393385, 0.4249448, -0.2873366, 0.4820393, -0.7213778, 0.7122814
3: -0.2396817, 0.3205543, -0.2754628, 0.3620429, -0.6017246, 0.5960170
4: -0.3388830, 0.3058940, -0.3944121, 0.3733480, -0.7122310, 0.7003061
5: -0.3658661, 0.4785203, -0.4207699, 0.5404676, -0.9063337, 0.8992902
6: -0.2650444, 0.3553236, -0.3079351, 0.4178966, -0.6829410, 0.6632587
7: -0.3520835, 0.3657104, -0.3978750, 0.4203667, -0.7724503, 0.7635854
8: -0.3408940, 0.4424144, -0.4044373, 0.5066215, -0.8475155, 0.8468517
9: -0.3346326, 0.4303319, -0.3901654, 0.4896750, -0.8243076, 0.8204974

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7925177, upper bound: 0.7908825
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7925177, upper bound: 0.7908825
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.3530945, 1.0728551, 0.2028129, 1.0957785, -0.7426839, 0.8700423
1: -0.3125902, 0.3045264, -0.4060443, 0.4025286, -0.7151189, 0.7105707
2: -0.2246469, 0.4060288, -0.3111620, 0.5116174, -0.7362642, 0.7171909
3: -0.2275940, 0.3054843, -0.2926248, 0.3897397, -0.6173337, 0.5981091
4: -0.3235123, 0.2883705, -0.4247174, 0.4162807, -0.7397929, 0.7130879
5: -0.3486819, 0.4561352, -0.4505081, 0.5720043, -0.9206862, 0.9066433
6: -0.2504518, 0.3380812, -0.3338593, 0.4476380, -0.6980898, 0.6719404
7: -0.3364048, 0.3472523, -0.4243075, 0.4482103, -0.7846152, 0.7715598
8: -0.3242080, 0.4210909, -0.4416491, 0.5393125, -0.8635205, 0.8627400
9: -0.3176938, 0.4087698, -0.4199692, 0.5200016, -0.8376955, 0.8287390

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7874587, upper bound: 0.7828662
time: 1.48 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7839765, upper bound: 0.7828662
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: 0.3450854, 1.0739660, 0.0685196, 1.1126741, -0.7675887, 1.0054464
1: -0.3172624, 0.3089916, -0.4989128, 0.4929237, -0.8101860, 0.8079044
2: -0.2287777, 0.4113258, -0.3852040, 0.5912036, -0.8199813, 0.7965298
3: -0.2309927, 0.3096809, -0.3440599, 0.4983316, -0.7293243, 0.6537408
4: -0.3278340, 0.2932526, -0.5098630, 0.5401040, -0.8679380, 0.8031156
5: -0.3535138, 0.4623984, -0.5604199, 0.6600924, -1.0136063, 1.0228183
6: -0.2545337, 0.3429292, -0.4050857, 0.5517133, -0.8062471, 0.7480148
7: -0.3408131, 0.3523504, -0.4951873, 0.5427332, -0.8835463, 0.8475378
8: -0.3287777, 0.4270866, -0.5445178, 0.6316414, -0.9604191, 0.9716043
9: -0.3224426, 0.4148325, -0.5045514, 0.6153742, -0.9378167, 0.9193838

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7924309, upper bound: 0.7908825
time: 1.80 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7924309, upper bound: 0.7912376
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.3120683, 1.0788932, 0.1102565, 1.1073194, -0.7952511, 0.9686368
1: -0.3372357, 0.3278563, -0.4695447, 0.4643153, -0.8015510, 0.7974010
2: -0.2467339, 0.4335788, -0.3618226, 0.5660307, -0.8127646, 0.7954013
3: -0.2451270, 0.3270723, -0.3279273, 0.4639841, -0.7091111, 0.6549996
4: -0.3465969, 0.3153112, -0.4829448, 0.5008879, -0.8474847, 0.7982560
5: -0.3742081, 0.4882120, -0.5258783, 0.6323832, -1.0065912, 1.0140903
6: -0.2713640, 0.3640856, -0.3825118, 0.5194765, -0.7908406, 0.7465974
7: -0.3590405, 0.3740931, -0.4727526, 0.5128678, -0.8719083, 0.8468457
8: -0.3494780, 0.4523564, -0.5121610, 0.6026054, -0.9520834, 0.9645175
9: -0.3428765, 0.4396604, -0.4777731, 0.5851949, -0.9280714, 0.9174335

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7838832, upper bound: 0.7855563
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7838832, upper bound: 0.7828662
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.2425796, 1.0893581, 0.3246102, 1.0769563, -0.8343767, 0.7647479
1: -0.3780392, 0.3726310, -0.3294400, 0.3204069, -0.6984461, 0.7020710
2: -0.2873366, 0.4820393, -0.2393385, 0.4249448, -0.7122814, 0.7213778
3: -0.2754628, 0.3620429, -0.2396817, 0.3205543, -0.5960170, 0.6017246
4: -0.3944121, 0.3733480, -0.3388830, 0.3058940, -0.7003061, 0.7122310
5: -0.4207699, 0.5404676, -0.3658661, 0.4785203, -0.8992902, 0.9063337
6: -0.3079351, 0.4178966, -0.2650444, 0.3553236, -0.6632587, 0.6829410
7: -0.3978750, 0.4203667, -0.3520835, 0.3657104, -0.7635854, 0.7724503
8: -0.4044373, 0.5066215, -0.3408940, 0.4424144, -0.8468517, 0.8475155
9: -0.3901654, 0.4896750, -0.3346326, 0.4303319, -0.8204974, 0.8243076

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7925177
time: 1.48 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7927590
time: 1.39 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.2028129, 1.0957785, 0.3530945, 1.0728551, -0.8700423, 0.7426839
1: -0.4060443, 0.4025286, -0.3125902, 0.3045264, -0.7105707, 0.7151189
2: -0.3111620, 0.5116174, -0.2246469, 0.4060288, -0.7171909, 0.7362642
3: -0.2926248, 0.3897397, -0.2275940, 0.3054843, -0.5981091, 0.6173337
4: -0.4247174, 0.4162807, -0.3235123, 0.2883705, -0.7130879, 0.7397929
5: -0.4505081, 0.5720043, -0.3486819, 0.4561352, -0.9066433, 0.9206862
6: -0.3338593, 0.4476380, -0.2504518, 0.3380812, -0.6719404, 0.6980898
7: -0.4243075, 0.4482103, -0.3364048, 0.3472523, -0.7715598, 0.7846152
8: -0.4416491, 0.5393125, -0.3242080, 0.4210909, -0.8627400, 0.8635205
9: -0.4199692, 0.5200016, -0.3176938, 0.4087698, -0.8287390, 0.8376955

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7874587
time: 1.62 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7839765
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.2425796, 1.0893581, 0.1840029, 1.0987369, -0.8561573, 0.9053552
1: -0.3780392, 0.3726310, -0.4143402, 0.4108763, -0.7889155, 0.7869712
2: -0.2873366, 0.4820393, -0.3233736, 0.5251237, -0.8124603, 0.8054129
3: -0.2754628, 0.3620429, -0.3020474, 0.3918061, -0.6672689, 0.6640903
4: -0.3944121, 0.3733480, -0.4369877, 0.4105251, -0.8049372, 0.8103357
5: -0.4207699, 0.5404676, -0.4615013, 0.5855693, -1.0063392, 1.0019689
6: -0.3079351, 0.4178966, -0.3348240, 0.4627579, -0.7706930, 0.7527206
7: -0.3978750, 0.4203667, -0.4292265, 0.4614596, -0.8593346, 0.8495932
8: -0.4044373, 0.5066215, -0.4465668, 0.5543190, -0.9587563, 0.9531883
9: -0.3901654, 0.4896750, -0.4302959, 0.5343053, -0.9244708, 0.9199709

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7909758
time: 7.81 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7913886
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.2028129, 1.0957785, 0.2163012, 1.0937321, -0.8909193, 0.8794773
1: -0.4060443, 0.4025286, -0.3948942, 0.3899386, -0.7959829, 0.7974228
2: -0.3111620, 0.5116174, -0.3040449, 0.5020361, -0.8131981, 0.8156623
3: -0.2926248, 0.3897397, -0.2876920, 0.3754800, -0.6681048, 0.6774317
4: -0.4247174, 0.4162807, -0.4141909, 0.3865119, -0.8112293, 0.8304716
5: -0.4505081, 0.5720043, -0.4394858, 0.5610155, -1.0115236, 1.0114900
6: -0.3338593, 0.4476380, -0.3188192, 0.4378725, -0.7717319, 0.7664572
7: -0.4243075, 0.4482103, -0.4115255, 0.4394256, -0.8637331, 0.8597358
8: -0.4416491, 0.5393125, -0.4220807, 0.5286034, -0.9702525, 0.9613932
9: -0.4199692, 0.5200016, -0.4082483, 0.5104357, -0.9304049, 0.9282500

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7856987
time: 1.79 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7830123
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0685196, 1.1126741, 0.3450854, 1.0739660, -1.0054464, 0.7675887
1: -0.4989128, 0.4929237, -0.3172624, 0.3089916, -0.8079044, 0.8101860
2: -0.3852040, 0.5912036, -0.2287777, 0.4113258, -0.7965298, 0.8199813
3: -0.3440599, 0.4983316, -0.2309927, 0.3096809, -0.6537408, 0.7293243
4: -0.5098630, 0.5401040, -0.3278340, 0.2932526, -0.8031156, 0.8679380
5: -0.5604199, 0.6600924, -0.3535138, 0.4623984, -1.0228183, 1.0136063
6: -0.4050857, 0.5517133, -0.2545337, 0.3429292, -0.7480148, 0.8062471
7: -0.4951873, 0.5427332, -0.3408131, 0.3523504, -0.8475378, 0.8835463
8: -0.5445178, 0.6316414, -0.3287777, 0.4270866, -0.9716043, 0.9604191
9: -0.5045514, 0.6153742, -0.3224426, 0.4148325, -0.9193838, 0.9378167

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7924309
time: 1.38 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7924309
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.1102565, 1.1073194, 0.3120683, 1.0788932, -0.9686368, 0.7952511
1: -0.4695447, 0.4643153, -0.3372357, 0.3278563, -0.7974010, 0.8015510
2: -0.3618226, 0.5660307, -0.2467339, 0.4335788, -0.7954013, 0.8127646
3: -0.3279273, 0.4639841, -0.2451270, 0.3270723, -0.6549996, 0.7091111
4: -0.4829448, 0.5008879, -0.3465969, 0.3153112, -0.7982560, 0.8474847
5: -0.5258783, 0.6323832, -0.3742081, 0.4882120, -1.0140903, 1.0065912
6: -0.3825118, 0.5194765, -0.2713640, 0.3640856, -0.7465974, 0.7908406
7: -0.4727526, 0.5128678, -0.3590405, 0.3740931, -0.8468457, 0.8719083
8: -0.5121610, 0.6026054, -0.3494780, 0.4523564, -0.9645175, 0.9520834
9: -0.4777731, 0.5851949, -0.3428765, 0.4396604, -0.9174335, 0.9280714

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7855563, upper bound: 0.7838832
time: 1.52 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7838832
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0685196, 1.1126741, 0.2055284, 1.0953846, -1.0268650, 0.9071457
1: -0.4989128, 0.4929237, -0.4039039, 0.4006516, -0.8995644, 0.8968276
2: -0.3852040, 0.5912036, -0.3096740, 0.5097811, -0.8949851, 0.9008775
3: -0.3440599, 0.4983316, -0.2915521, 0.3872339, -0.7312938, 0.7898837
4: -0.5098630, 0.5401040, -0.4227579, 0.4136398, -0.9235028, 0.9628619
5: -0.5604199, 0.6600924, -0.4483720, 0.5700287, -1.1304486, 1.1084645
6: -0.4050857, 0.5517133, -0.3322661, 0.4455262, -0.8506119, 0.8839794
7: -0.4951873, 0.5427332, -0.4226661, 0.4463062, -0.9414935, 0.9653993
8: -0.5445178, 0.6316414, -0.4392927, 0.5372761, -1.0817939, 1.0709341
9: -0.5045514, 0.6153742, -0.4180084, 0.5181274, -1.0226789, 1.0333827

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7908825
time: 1.45 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7908825
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.1102565, 1.1073194, 0.1664318, 1.1008693, -0.9906129, 0.9408876
1: -0.4695447, 0.4643153, -0.4337140, 0.4295704, -0.8991151, 0.8980293
2: -0.3618226, 0.5660307, -0.3330629, 0.5353544, -0.8971770, 0.8990936
3: -0.3279273, 0.4639841, -0.3074898, 0.4221312, -0.7500585, 0.7714739
4: -0.4829448, 0.5008879, -0.4500493, 0.4534594, -0.9364042, 0.9509372
5: -0.5258783, 0.6323832, -0.4826078, 0.5975422, -1.1234205, 1.1149909
6: -0.3825118, 0.5194765, -0.3553208, 0.4749389, -0.8574507, 0.8747973
7: -0.4727526, 0.5128678, -0.4455230, 0.4762526, -0.9490052, 0.9583908
8: -0.5121610, 0.6026054, -0.4721132, 0.5660451, -1.0782061, 1.0747186
9: -0.4777731, 0.5851949, -0.4453186, 0.5485135, -1.0262866, 1.0305135

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7855563, upper bound: 0.7828662
time: 1.45 seconds

## Relational analysis of NS_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7828662
time: 1.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.61 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7925177, upper bound: 0.7924309
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7925177, upper bound: 0.7924309
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7874587, upper bound: 0.7838832
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7839765, upper bound: 0.7838832
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7924309, upper bound: 0.7924309
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7924309, upper bound: 0.7926701
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7838832, upper bound: 0.7872906
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7838832, upper bound: 0.7838832
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7925177, upper bound: 0.7908825
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7925177, upper bound: 0.7908825
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7874587, upper bound: 0.7828662
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7839765, upper bound: 0.7828662
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7924309, upper bound: 0.7908825
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7924309, upper bound: 0.7912376
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7838832, upper bound: 0.7855563
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7838832, upper bound: 0.7828662
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7925177
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7927590
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7874587
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7839765
NS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7909758
NS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7913886
NS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7856987
NS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7830123
NS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7924309
NS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7924309
NS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7855563, upper bound: 0.7838832
NS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7838832
NS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7908825
NS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7908825, upper bound: 0.7908825
NS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7855563, upper bound: 0.7828662
NS_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 0, lower bound: -0.7828662, upper bound: 0.7828662

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.4513598, 1.0592251, 0.3810282, 1.0689805, -0.6176206, 0.6781969
1: -0.2552677, 0.2497412, -0.2962954, 0.2889529, -0.5442206, 0.5460365
2: -0.1739639, 0.3410399, -0.2102395, 0.3875544, -0.5615184, 0.5512794
3: -0.1858937, 0.2539964, -0.2157398, 0.2908480, -0.4767417, 0.4697363
4: -0.2704857, 0.2284714, -0.3084385, 0.2713432, -0.5418289, 0.5369098
5: -0.2893996, 0.3792900, -0.3318299, 0.4342905, -0.7236901, 0.7111199
6: -0.2003706, 0.2785980, -0.2362154, 0.3211720, -0.5215425, 0.5148134
7: -0.2823159, 0.2847014, -0.3210292, 0.3294711, -0.6117871, 0.6057305
8: -0.2681431, 0.3475285, -0.3082706, 0.4001795, -0.6683227, 0.6557990
9: -0.2594296, 0.3343846, -0.3011312, 0.3876245, -0.6470541, 0.6355158

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7872462, upper bound: 0.7835745
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7835745
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.4195517, 1.0636371, 0.3810282, 1.0689805, -0.6494288, 0.6826090
1: -0.2738228, 0.2674749, -0.2962954, 0.2889529, -0.5627757, 0.5637703
2: -0.1903698, 0.3620768, -0.2102395, 0.3875544, -0.5779243, 0.5723162
3: -0.1993918, 0.2706630, -0.2157398, 0.2908480, -0.4902398, 0.4864028
4: -0.2876503, 0.2478605, -0.3084385, 0.2713432, -0.5589935, 0.5562990
5: -0.3085891, 0.4041645, -0.3318299, 0.4342905, -0.7428796, 0.7359944
6: -0.2165818, 0.2978524, -0.2362154, 0.3211720, -0.5377538, 0.5340678
7: -0.2998243, 0.3049489, -0.3210292, 0.3294711, -0.6292955, 0.6259780
8: -0.2862911, 0.3713405, -0.3082706, 0.4001795, -0.6864706, 0.6796111
9: -0.2782895, 0.3584630, -0.3011312, 0.3876245, -0.6659140, 0.6595942

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7881663
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7835745
time: 1.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.4647607, 1.0573665, 0.3432153, 1.0742253, -0.6094646, 0.7141512
1: -0.2474504, 0.2422698, -0.3183533, 0.3100343, -0.5574847, 0.5606231
2: -0.1670521, 0.3321773, -0.2297424, 0.4125623, -0.5796145, 0.5619198
3: -0.1802068, 0.2469749, -0.2317863, 0.3106607, -0.4908675, 0.4787612
4: -0.2632543, 0.2203026, -0.3288432, 0.2943927, -0.5576470, 0.5491458
5: -0.2813150, 0.3688104, -0.3546419, 0.4638610, -0.7451760, 0.7234523
6: -0.1935408, 0.2704861, -0.2554868, 0.3440613, -0.5376022, 0.5259729
7: -0.2749396, 0.2761710, -0.3418426, 0.3535409, -0.6284806, 0.6180136
8: -0.2604974, 0.3374965, -0.3298445, 0.4284866, -0.6889839, 0.6673410
9: -0.2514840, 0.3242405, -0.3235514, 0.4162482, -0.6677321, 0.6477919

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7816130, upper bound: 0.7723215
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7819102, upper bound: 0.7785256
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.4779240, 1.0555407, 0.3738099, 1.0699816, -0.5920576, 0.6817307
1: -0.2397717, 0.2349310, -0.3005061, 0.2929769, -0.5327486, 0.5354371
2: -0.1602628, 0.3234717, -0.2139623, 0.3923284, -0.5525911, 0.5374340
3: -0.1746208, 0.2400777, -0.2188030, 0.2946301, -0.4692509, 0.4588807
4: -0.2561511, 0.2122787, -0.3123336, 0.2757430, -0.5318941, 0.5246123
5: -0.2733737, 0.3585163, -0.3361844, 0.4399354, -0.7133090, 0.6947007
6: -0.1868321, 0.2625178, -0.2398941, 0.3255414, -0.5123735, 0.5024119
7: -0.2676941, 0.2677918, -0.3250021, 0.3340657, -0.6017599, 0.5927939
8: -0.2529871, 0.3276424, -0.3123888, 0.4055831, -0.6585702, 0.6400312
9: -0.2436791, 0.3142760, -0.3054110, 0.3930883, -0.6367674, 0.6196870

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7723215
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7785256
time: 1.50 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.3450854, 1.0739660, 0.3783439, 1.0690706, -0.7239852, 0.6956222
1: -0.3172624, 0.3089916, -0.2968305, 0.2894124, -0.6066748, 0.6058221
2: -0.2287777, 0.4113258, -0.2107787, 0.3881055, -0.6168833, 0.6221045
3: -0.2309927, 0.3096809, -0.2164033, 0.2913524, -0.5223451, 0.5260842
4: -0.3278340, 0.2932526, -0.3089506, 0.2718954, -0.5997294, 0.6022032
5: -0.3535138, 0.4623984, -0.3328621, 0.4353005, -0.7888144, 0.7952604
6: -0.2545337, 0.3429292, -0.2365461, 0.3234731, -0.5780069, 0.5794753
7: -0.3408131, 0.3523504, -0.3214570, 0.3301038, -0.6709169, 0.6738074
8: -0.3287777, 0.4270866, -0.3091323, 0.4012094, -0.7299870, 0.7362188
9: -0.3224426, 0.4148325, -0.3015842, 0.3882285, -0.7106711, 0.7164167

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7881663, upper bound: 0.7832194
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7832194
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.3450854, 1.0739660, 0.3426425, 1.0743048, -0.7292194, 0.7313235
1: -0.3172624, 0.3089916, -0.3186874, 0.3103536, -0.6276159, 0.6276790
2: -0.2287777, 0.4113258, -0.2300377, 0.4129413, -0.6417190, 0.6413635
3: -0.2309927, 0.3096809, -0.2320295, 0.3109608, -0.5419536, 0.5417104
4: -0.3278340, 0.2932526, -0.3291523, 0.2947418, -0.6225758, 0.6224049
5: -0.3535138, 0.4623984, -0.3549875, 0.4643087, -0.8178226, 0.8173858
6: -0.2545337, 0.3429292, -0.2557788, 0.3444080, -0.5989417, 0.5987080
7: -0.3408131, 0.3523504, -0.3421579, 0.3539055, -0.6947186, 0.6945083
8: -0.3287777, 0.4270866, -0.3301713, 0.4289153, -0.7576929, 0.7572578
9: -0.3224426, 0.4148325, -0.3238910, 0.4166818, -0.7391244, 0.7387235

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7881663, upper bound: 0.7838832
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7838832
time: 1.34 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.3120683, 1.0788932, 0.3844179, 1.0676370, -0.7555687, 0.6944753
1: -0.3372357, 0.3278563, -0.2911284, 0.2838541, -0.6210898, 0.6189847
2: -0.2467339, 0.4335788, -0.2058740, 0.3815243, -0.6282582, 0.6394527
3: -0.2451270, 0.3270723, -0.2128418, 0.2862799, -0.5314069, 0.5399141
4: -0.3465969, 0.3153112, -0.3037116, 0.2659438, -0.6125407, 0.6190228
5: -0.3742081, 0.4882120, -0.3279903, 0.4282605, -0.8024686, 0.8162023
6: -0.2713640, 0.3640856, -0.2312792, 0.3211957, -0.5925597, 0.5953648
7: -0.3590405, 0.3740931, -0.3159164, 0.3239816, -0.6830221, 0.6900095
8: -0.3494780, 0.4523564, -0.3042923, 0.3946040, -0.7440820, 0.7566487
9: -0.3428765, 0.4396604, -0.2955984, 0.3806399, -0.7235165, 0.7352589

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7766175
time: 1.70 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7817862
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.3428528, 1.0742756, 0.3918149, 1.0655868, -0.7227340, 0.6824607
1: -0.3185647, 0.3102363, -0.2830655, 0.2759669, -0.5945317, 0.5933018
2: -0.2299292, 0.4128022, -0.1989914, 0.3721859, -0.6021152, 0.6117936
3: -0.2319401, 0.3108507, -0.2080588, 0.2791265, -0.5110666, 0.5189095
4: -0.3290388, 0.2946135, -0.2963151, 0.2576436, -0.5866824, 0.5909286
5: -0.3548607, 0.4641442, -0.3215500, 0.4185370, -0.7733977, 0.7856942
6: -0.2556715, 0.3442807, -0.2237457, 0.3193280, -0.5749995, 0.5680264
7: -0.3420421, 0.3537716, -0.3080338, 0.3153698, -0.6574119, 0.6618054
8: -0.3300514, 0.4287578, -0.2978347, 0.3855072, -0.7155586, 0.7265924
9: -0.3237663, 0.4165224, -0.2870816, 0.3698567, -0.6936229, 0.7036040

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7723215
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7785256
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.4513598, 1.0592251, 0.2425796, 1.0893581, -0.6379983, 0.8166455
1: -0.2552677, 0.2497412, -0.3780392, 0.3726310, -0.6278987, 0.6277803
2: -0.1739639, 0.3410399, -0.2873366, 0.4820393, -0.6560033, 0.6283765
3: -0.1858937, 0.2539964, -0.2754628, 0.3620429, -0.5479366, 0.5294592
4: -0.2704857, 0.2284714, -0.3944121, 0.3733480, -0.6438337, 0.6228836
5: -0.2893996, 0.3792900, -0.4207699, 0.5404676, -0.8298672, 0.8000599
6: -0.2003706, 0.2785980, -0.3079351, 0.4178966, -0.6182672, 0.5865332
7: -0.2823159, 0.2847014, -0.3978750, 0.4203667, -0.7026826, 0.6825764
8: -0.2681431, 0.3475285, -0.4044373, 0.5066215, -0.7747646, 0.7519658
9: -0.2594296, 0.3343846, -0.3901654, 0.4896750, -0.7491046, 0.7245501

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7862511
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7824318
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.4195517, 1.0636371, 0.2425796, 1.0893581, -0.6698064, 0.8210576
1: -0.2738228, 0.2674749, -0.3780392, 0.3726310, -0.6464538, 0.6455141
2: -0.1903698, 0.3620768, -0.2873366, 0.4820393, -0.6724092, 0.6494133
3: -0.1993918, 0.2706630, -0.2754628, 0.3620429, -0.5614347, 0.5461258
4: -0.2876503, 0.2478605, -0.3944121, 0.3733480, -0.6609983, 0.6422726
5: -0.3085891, 0.4041645, -0.4207699, 0.5404676, -0.8490567, 0.8249344
6: -0.2165818, 0.2978524, -0.3079351, 0.4178966, -0.6344784, 0.6057876
7: -0.2998243, 0.3049489, -0.3978750, 0.4203667, -0.7201911, 0.7028238
8: -0.2862911, 0.3713405, -0.4044373, 0.5066215, -0.7929126, 0.7757778
9: -0.2782895, 0.3584630, -0.3901654, 0.4896750, -0.7679645, 0.7486284

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7862511
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7824318
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.4647607, 1.0573665, 0.2028129, 1.0957785, -0.6310178, 0.8545536
1: -0.2474504, 0.2422698, -0.4060443, 0.4025286, -0.6499791, 0.6483141
2: -0.1670521, 0.3321773, -0.3111620, 0.5116174, -0.6786695, 0.6433393
3: -0.1802068, 0.2469749, -0.2926248, 0.3897397, -0.5699465, 0.5395997
4: -0.2632543, 0.2203026, -0.4247174, 0.4162807, -0.6795350, 0.6450200
5: -0.2813150, 0.3688104, -0.4505081, 0.5720043, -0.8533192, 0.8193184
6: -0.1935408, 0.2704861, -0.3338593, 0.4476380, -0.6411788, 0.6043454
7: -0.2749396, 0.2761710, -0.4243075, 0.4482103, -0.7231500, 0.7004786
8: -0.2604974, 0.3374965, -0.4416491, 0.5393125, -0.7998098, 0.7791456
9: -0.2514840, 0.3242405, -0.4199692, 0.5200016, -0.7714856, 0.7442098

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7814502, upper bound: 0.7714915
time: 1.79 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7819102, upper bound: 0.7776602
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.4779240, 1.0555407, 0.2409854, 1.0899091, -0.6119851, 0.8145553
1: -0.2397717, 0.2349310, -0.3800657, 0.3748972, -0.6146689, 0.6149968
2: -0.1602628, 0.3234717, -0.2892219, 0.4843833, -0.6446460, 0.6126936
3: -0.1746208, 0.2400777, -0.2766636, 0.3637355, -0.5383563, 0.5167413
4: -0.2561511, 0.2122787, -0.3967031, 0.3770086, -0.6331596, 0.6089818
5: -0.2733737, 0.3585163, -0.4226792, 0.5426787, -0.8160524, 0.7811955
6: -0.1868321, 0.2625178, -0.3101559, 0.4187550, -0.6055871, 0.5726738
7: -0.2676941, 0.2677918, -0.3999803, 0.4224984, -0.6901926, 0.6677721
8: -0.2529871, 0.3276424, -0.4072005, 0.5089710, -0.7619581, 0.7348429
9: -0.2436791, 0.3142760, -0.3926101, 0.4921871, -0.7358662, 0.7068861

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7714915
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7776602
time: 1.39 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.3450854, 1.0739660, 0.2397356, 1.0894573, -0.7443719, 0.8342304
1: -0.3172624, 0.3089916, -0.3785933, 0.3731779, -0.6904402, 0.6875849
2: -0.2287777, 0.4113258, -0.2879410, 0.4826305, -0.7114083, 0.6992667
3: -0.2309927, 0.3096809, -0.2761283, 0.3625638, -0.5935565, 0.5858091
4: -0.3278340, 0.2932526, -0.3950387, 0.3741408, -0.7019747, 0.6882913
5: -0.3535138, 0.4623984, -0.4218369, 0.5415114, -0.8950253, 0.8842353
6: -0.2545337, 0.3429292, -0.3083424, 0.4205120, -0.6750457, 0.6512715
7: -0.3408131, 0.3523504, -0.3983532, 0.4210410, -0.7618542, 0.7507036
8: -0.3287777, 0.4270866, -0.4054458, 0.5077188, -0.8364965, 0.8325323
9: -0.3224426, 0.4148325, -0.3907088, 0.4902700, -0.8127126, 0.8055413

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7853286
time: 1.40 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7821339
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.3450854, 1.0739660, 0.2020860, 1.0958800, -0.7507946, 0.8718800
1: -0.3172624, 0.3089916, -0.4065972, 0.4030688, -0.7203312, 0.7155887
2: -0.2287777, 0.4113258, -0.3115996, 0.5120916, -0.7408694, 0.7229253
3: -0.2309927, 0.3096809, -0.2929218, 0.3903867, -0.6213794, 0.6026027
4: -0.3278340, 0.2932526, -0.4252234, 0.4170234, -0.7448574, 0.7184761
5: -0.3535138, 0.4623984, -0.4511495, 0.5725144, -0.9260283, 0.9135478
6: -0.2545337, 0.3429292, -0.3342881, 0.4481832, -0.7027169, 0.6772172
7: -0.3408131, 0.3523504, -0.4247312, 0.4487706, -0.7895837, 0.7770816
8: -0.3287777, 0.4270866, -0.4422578, 0.5398465, -0.8686242, 0.8693444
9: -0.3224426, 0.4148325, -0.4204756, 0.5205714, -0.8430139, 0.8353081

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7881660, upper bound: 0.7828662
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7828662
time: 1.38 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.3120683, 1.0788932, 0.2432680, 1.0882728, -0.7762045, 0.8356252
1: -0.3372357, 0.3278563, -0.3743070, 0.3681074, -0.7053431, 0.7021633
2: -0.2467339, 0.4335788, -0.2837795, 0.4773868, -0.7241207, 0.7173582
3: -0.2451270, 0.3270723, -0.2734602, 0.3587965, -0.6039234, 0.6005325
4: -0.3465969, 0.3153112, -0.3899705, 0.3659808, -0.7125777, 0.7052816
5: -0.3742081, 0.4882120, -0.4178367, 0.5365572, -0.9107653, 0.9060487
6: -0.2713640, 0.3640856, -0.3034816, 0.4186043, -0.6899683, 0.6675672
7: -0.3590405, 0.3740931, -0.3936996, 0.4162704, -0.7753110, 0.7677928
8: -0.3494780, 0.4523564, -0.3993132, 0.5027325, -0.8522105, 0.8516696
9: -0.3428765, 0.4396604, -0.3852395, 0.4848485, -0.8277251, 0.8249000

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7750172
time: 1.76 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7801889
time: 1.38 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.3428528, 1.0742756, 0.2533073, 1.0856249, -0.7427722, 0.8209683
1: -0.3185647, 0.3102363, -0.3645510, 0.3567545, -0.6753193, 0.6747873
2: -0.2299292, 0.4128022, -0.2742728, 0.4655395, -0.6954687, 0.6870750
3: -0.2319401, 0.3108507, -0.2671825, 0.3502576, -0.5821977, 0.5780332
4: -0.3290388, 0.2946135, -0.3784614, 0.3484789, -0.6775178, 0.6730750
5: -0.3548607, 0.4641442, -0.4083852, 0.5250597, -0.8799204, 0.8725294
6: -0.2556715, 0.3442807, -0.2928921, 0.4123787, -0.6680502, 0.6371728
7: -0.3420421, 0.3537716, -0.3833753, 0.4053868, -0.7474290, 0.7371469
8: -0.3300514, 0.4287578, -0.3856735, 0.4910315, -0.8210829, 0.8144312
9: -0.3237663, 0.4165224, -0.3730465, 0.4726087, -0.7963750, 0.7895689

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7714915
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7776602
time: 1.38 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.2425796, 1.0893581, 0.4513598, 1.0592251, -0.8166455, 0.6379983
1: -0.3780392, 0.3726310, -0.2552677, 0.2497412, -0.6277803, 0.6278987
2: -0.2873366, 0.4820393, -0.1739639, 0.3410399, -0.6283765, 0.6560033
3: -0.2754628, 0.3620429, -0.1858937, 0.2539964, -0.5294592, 0.5479366
4: -0.3944121, 0.3733480, -0.2704857, 0.2284714, -0.6228836, 0.6438337
5: -0.4207699, 0.5404676, -0.2893996, 0.3792900, -0.8000599, 0.8298672
6: -0.3079351, 0.4178966, -0.2003706, 0.2785980, -0.5865332, 0.6182672
7: -0.3978750, 0.4203667, -0.2823159, 0.2847014, -0.6825764, 0.7026826
8: -0.4044373, 0.5066215, -0.2681431, 0.3475285, -0.7519658, 0.7747646
9: -0.3901654, 0.4896750, -0.2594296, 0.3343846, -0.7245501, 0.7491046

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862511, upper bound: 0.7832750
time: 1.85 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7832750
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.2425796, 1.0893581, 0.4195517, 1.0636371, -0.8210576, 0.6698064
1: -0.3780392, 0.3726310, -0.2738228, 0.2674749, -0.6455141, 0.6464538
2: -0.2873366, 0.4820393, -0.1903698, 0.3620768, -0.6494133, 0.6724092
3: -0.2754628, 0.3620429, -0.1993918, 0.2706630, -0.5461258, 0.5614347
4: -0.3944121, 0.3733480, -0.2876503, 0.2478605, -0.6422726, 0.6609983
5: -0.4207699, 0.5404676, -0.3085891, 0.4041645, -0.8249344, 0.8490567
6: -0.3079351, 0.4178966, -0.2165818, 0.2978524, -0.6057876, 0.6344784
7: -0.3978750, 0.4203667, -0.2998243, 0.3049489, -0.7028238, 0.7201911
8: -0.4044373, 0.5066215, -0.2862911, 0.3713405, -0.7757778, 0.7929126
9: -0.3901654, 0.4896750, -0.2782895, 0.3584630, -0.7486284, 0.7679645

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862511, upper bound: 0.7839765
time: 1.81 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7839765
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.2028129, 1.0957785, 0.4647607, 1.0573665, -0.8545536, 0.6310178
1: -0.4060443, 0.4025286, -0.2474504, 0.2422698, -0.6483141, 0.6499791
2: -0.3111620, 0.5116174, -0.1670521, 0.3321773, -0.6433393, 0.6786695
3: -0.2926248, 0.3897397, -0.1802068, 0.2469749, -0.5395997, 0.5699465
4: -0.4247174, 0.4162807, -0.2632543, 0.2203026, -0.6450200, 0.6795350
5: -0.4505081, 0.5720043, -0.2813150, 0.3688104, -0.8193184, 0.8533192
6: -0.3338593, 0.4476380, -0.1935408, 0.2704861, -0.6043454, 0.6411788
7: -0.4243075, 0.4482103, -0.2749396, 0.2761710, -0.7004786, 0.7231500
8: -0.4416491, 0.5393125, -0.2604974, 0.3374965, -0.7791456, 0.7998098
9: -0.4199692, 0.5200016, -0.2514840, 0.3242405, -0.7442098, 0.7714856

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7814502
time: 1.54 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7819102
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.2409854, 1.0899091, 0.4779240, 1.0555407, -0.8145553, 0.6119851
1: -0.3800657, 0.3748972, -0.2397717, 0.2349310, -0.6149968, 0.6146689
2: -0.2892219, 0.4843833, -0.1602628, 0.3234717, -0.6126936, 0.6446460
3: -0.2766636, 0.3637355, -0.1746208, 0.2400777, -0.5167413, 0.5383563
4: -0.3967031, 0.3770086, -0.2561511, 0.2122787, -0.6089818, 0.6331596
5: -0.4226792, 0.5426787, -0.2733737, 0.3585163, -0.7811955, 0.8160524
6: -0.3101559, 0.4187550, -0.1868321, 0.2625178, -0.5726738, 0.6055871
7: -0.3999803, 0.4224984, -0.2676941, 0.2677918, -0.6677721, 0.6901926
8: -0.4072005, 0.5089710, -0.2529871, 0.3276424, -0.7348429, 0.7619581
9: -0.3926101, 0.4921871, -0.2436791, 0.3142760, -0.7068861, 0.7358662

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7786506
time: 1.67 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7786506
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.2425796, 1.0893581, 0.3215611, 1.0774264, -0.8348469, 0.7677970
1: -0.3780392, 0.3726310, -0.3313494, 0.3220886, -0.7001278, 0.7039803
2: -0.2873366, 0.4820393, -0.2410323, 0.4269665, -0.7143031, 0.7230716
3: -0.2754628, 0.3620429, -0.2409701, 0.3221849, -0.5976477, 0.6030130
4: -0.3944121, 0.3733480, -0.3405106, 0.3080422, -0.7024543, 0.7138586
5: -0.4207699, 0.5404676, -0.3677861, 0.4809170, -0.9016870, 0.9082537
6: -0.3079351, 0.4178966, -0.2666077, 0.3572239, -0.6651591, 0.6845043
7: -0.3978750, 0.4203667, -0.3537761, 0.3677433, -0.7656182, 0.7741429
8: -0.4044373, 0.5066215, -0.3427653, 0.4447796, -0.8492169, 0.8493868
9: -0.3901654, 0.4896750, -0.3365375, 0.4326150, -0.8227805, 0.8262125

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7854322
time: 1.72 seconds

## Relational analysis of NS_A2_B1_B2_A1_B1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7822137
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.2425796, 1.0893581, 0.2889470, 1.0824758, -0.8398962, 0.8004110
1: -0.3780392, 0.3726310, -0.3511562, 0.3428452, -0.7208844, 0.7237872
2: -0.2873366, 0.4820393, -0.2605705, 0.4501067, -0.7374433, 0.7426099
3: -0.2754628, 0.3620429, -0.2554035, 0.3387596, -0.6142223, 0.6174464
4: -0.3944121, 0.3733480, -0.3629163, 0.3325011, -0.7269133, 0.7362643
5: -0.4207699, 0.5404676, -0.3899683, 0.5057890, -0.9265590, 0.9304359
6: -0.3079351, 0.4178966, -0.2828213, 0.3819000, -0.6898352, 0.7007179
7: -0.3978750, 0.4203667, -0.3717121, 0.3898664, -0.7877413, 0.7920789
8: -0.4044373, 0.5066215, -0.3670068, 0.4707648, -0.8752022, 0.8736283
9: -0.3901654, 0.4896750, -0.3586593, 0.4567480, -0.8469135, 0.8483343

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862511, upper bound: 0.7830123
time: 1.59 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7830123
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.2028129, 1.0957785, 0.3323683, 1.0757461, -0.8729332, 0.7634102
1: -0.4060443, 0.4025286, -0.3247063, 0.3160818, -0.7221261, 0.7272350
2: -0.3111620, 0.5116174, -0.2353370, 0.4197448, -0.7309068, 0.7469543
3: -0.2926248, 0.3897397, -0.2363894, 0.3163601, -0.6089849, 0.6261292
4: -0.4247174, 0.4162807, -0.3346967, 0.3010221, -0.7257395, 0.7509773
5: -0.4505081, 0.5720043, -0.3611860, 0.4723554, -0.9228635, 0.9331903
6: -0.3338593, 0.4476380, -0.2610232, 0.3506273, -0.6844866, 0.7086612
7: -0.4243075, 0.4482103, -0.3478132, 0.3604812, -0.7847887, 0.7960235
8: -0.4416491, 0.5393125, -0.3360806, 0.4366067, -0.8782558, 0.8753930
9: -0.4199692, 0.5200016, -0.3299884, 0.4244592, -0.8444284, 0.8499900

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7800391
time: 1.54 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7802552
time: 1.52 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.2397356, 1.0894573, 0.3450854, 1.0739660, -0.8342304, 0.7443719
1: -0.3785933, 0.3731779, -0.3172624, 0.3089916, -0.6875849, 0.6904402
2: -0.2879410, 0.4826305, -0.2287777, 0.4113258, -0.6992667, 0.7114083
3: -0.2761283, 0.3625638, -0.2309927, 0.3096809, -0.5858091, 0.5935565
4: -0.3950387, 0.3741408, -0.3278340, 0.2932526, -0.6882913, 0.7019747
5: -0.4218369, 0.5415114, -0.3535138, 0.4623984, -0.8842353, 0.8950253
6: -0.3083424, 0.4205120, -0.2545337, 0.3429292, -0.6512715, 0.6750457
7: -0.3983532, 0.4210410, -0.3408131, 0.3523504, -0.7507036, 0.7618542
8: -0.4054458, 0.5077188, -0.3287777, 0.4270866, -0.8325323, 0.8364965
9: -0.3907088, 0.4902700, -0.3224426, 0.4148325, -0.8055413, 0.8127126

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7853286, upper bound: 0.7835745
time: 1.67 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7835745
time: 1.61 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.2020860, 1.0958800, 0.3450854, 1.0739660, -0.8718800, 0.7507946
1: -0.4065972, 0.4030688, -0.3172624, 0.3089916, -0.7155887, 0.7203312
2: -0.3115996, 0.5120916, -0.2287777, 0.4113258, -0.7229253, 0.7408694
3: -0.2929218, 0.3903867, -0.2309927, 0.3096809, -0.6026027, 0.6213794
4: -0.4252234, 0.4170234, -0.3278340, 0.2932526, -0.7184761, 0.7448574
5: -0.4511495, 0.5725144, -0.3535138, 0.4623984, -0.9135478, 0.9260283
6: -0.3342881, 0.4481832, -0.2545337, 0.3429292, -0.6772172, 0.7027169
7: -0.4247312, 0.4487706, -0.3408131, 0.3523504, -0.7770816, 0.7895837
8: -0.4422578, 0.5398465, -0.3287777, 0.4270866, -0.8693444, 0.8686242
9: -0.4204756, 0.5205714, -0.3224426, 0.4148325, -0.8353081, 0.8430139

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7881660
time: 1.75 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7835745
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.2432680, 1.0882728, 0.3120683, 1.0788932, -0.8356252, 0.7762045
1: -0.3743070, 0.3681074, -0.3372357, 0.3278563, -0.7021633, 0.7053431
2: -0.2837795, 0.4773868, -0.2467339, 0.4335788, -0.7173582, 0.7241207
3: -0.2734602, 0.3587965, -0.2451270, 0.3270723, -0.6005325, 0.6039234
4: -0.3899705, 0.3659808, -0.3465969, 0.3153112, -0.7052816, 0.7125777
5: -0.4178367, 0.5365572, -0.3742081, 0.4882120, -0.9060487, 0.9107653
6: -0.3034816, 0.4186043, -0.2713640, 0.3640856, -0.6675672, 0.6899683
7: -0.3936996, 0.4162704, -0.3590405, 0.3740931, -0.7677928, 0.7753110
8: -0.3993132, 0.5027325, -0.3494780, 0.4523564, -0.8516696, 0.8522105
9: -0.3852395, 0.4848485, -0.3428765, 0.4396604, -0.8249000, 0.8277251

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7750172, upper bound: 0.7785256
time: 2.51 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7801889, upper bound: 0.7785256
time: 2.08 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.2533073, 1.0856249, 0.3428528, 1.0742756, -0.8209683, 0.7427722
1: -0.3645510, 0.3567545, -0.3185647, 0.3102363, -0.6747873, 0.6753193
2: -0.2742728, 0.4655395, -0.2299292, 0.4128022, -0.6870750, 0.6954687
3: -0.2671825, 0.3502576, -0.2319401, 0.3108507, -0.5780332, 0.5821977
4: -0.3784614, 0.3484789, -0.3290388, 0.2946135, -0.6730750, 0.6775178
5: -0.4083852, 0.5250597, -0.3548607, 0.4641442, -0.8725294, 0.8799204
6: -0.2928921, 0.4123787, -0.2556715, 0.3442807, -0.6371728, 0.6680502
7: -0.3833753, 0.4053868, -0.3420421, 0.3537716, -0.7371469, 0.7474290
8: -0.3856735, 0.4910315, -0.3300514, 0.4287578, -0.8144312, 0.8210829
9: -0.3730465, 0.4726087, -0.3237663, 0.4165224, -0.7895689, 0.7963750

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7785256
time: 1.62 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7785256
time: 1.63 seconds

## BFS NS instance: NS_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.2397356, 1.0894573, 0.2055284, 1.0953846, -0.8556490, 0.8839289
1: -0.3785933, 0.3731779, -0.4039039, 0.4006516, -0.7792449, 0.7770818
2: -0.2879410, 0.4826305, -0.3096740, 0.5097811, -0.7977221, 0.7923045
3: -0.2761283, 0.3625638, -0.2915521, 0.3872339, -0.6633621, 0.6541158
4: -0.3950387, 0.3741408, -0.4227579, 0.4136398, -0.8086784, 0.7968987
5: -0.4218369, 0.5415114, -0.4483720, 0.5700287, -0.9918655, 0.9898834
6: -0.3083424, 0.4205120, -0.3322661, 0.4455262, -0.7538686, 0.7527781
7: -0.3983532, 0.4210410, -0.4226661, 0.4463062, -0.8446593, 0.8437072
8: -0.4054458, 0.5077188, -0.4392927, 0.5372761, -0.9427218, 0.9470115
9: -0.3907088, 0.4902700, -0.4180084, 0.5181274, -0.9088362, 0.9082783

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7862510
time: 1.69 seconds

## Relational analysis of NS_A2_B2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7824318
time: 2.45 seconds

## BFS NS instance: NS_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.2020860, 1.0958800, 0.2055284, 1.0953846, -0.8932986, 0.8903517
1: -0.4065972, 0.4030688, -0.4039039, 0.4006516, -0.8072488, 0.8069727
2: -0.3115996, 0.5120916, -0.3096740, 0.5097811, -0.8213807, 0.8217656
3: -0.2929218, 0.3903867, -0.2915521, 0.3872339, -0.6801556, 0.6819388
4: -0.4252234, 0.4170234, -0.4227579, 0.4136398, -0.8388631, 0.8397813
5: -0.4511495, 0.5725144, -0.4483720, 0.5700287, -1.0211781, 1.0208864
6: -0.3342881, 0.4481832, -0.3322661, 0.4455262, -0.7798143, 0.7804493
7: -0.4247312, 0.4487706, -0.4226661, 0.4463062, -0.8710374, 0.8714367
8: -0.4422578, 0.5398465, -0.4392927, 0.5372761, -0.9795339, 0.9791392
9: -0.4204756, 0.5205714, -0.4180084, 0.5181274, -0.9386030, 0.9385797

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7862510
time: 1.72 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7824318
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.2432680, 1.0882728, 0.1664318, 1.1008693, -0.8576013, 0.9218410
1: -0.3743070, 0.3681074, -0.4337140, 0.4295704, -0.8038774, 0.8018215
2: -0.2837795, 0.4773868, -0.3330629, 0.5353544, -0.8191339, 0.8104497
3: -0.2734602, 0.3587965, -0.3074898, 0.4221312, -0.6955914, 0.6662863
4: -0.3899705, 0.3659808, -0.4500493, 0.4534594, -0.8434299, 0.8160301
5: -0.4178367, 0.5365572, -0.4826078, 0.5975422, -1.0153790, 1.0191650
6: -0.3034816, 0.4186043, -0.3553208, 0.4749389, -0.7784204, 0.7739251
7: -0.3936996, 0.4162704, -0.4455230, 0.4762526, -0.8699522, 0.8617934
8: -0.3993132, 0.5027325, -0.4721132, 0.5660451, -0.9653583, 0.9748457
9: -0.3852395, 0.4848485, -0.4453186, 0.5485135, -0.9337531, 0.9301671

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7750172, upper bound: 0.7776602
time: 1.74 seconds

## Relational analysis of NS_A2_B2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7801889, upper bound: 0.7776602
time: 1.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.81 seconds
NS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7872462, upper bound: 0.7835745
NS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7835745
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7881663
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7835745
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7816130, upper bound: 0.7723215
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7819102, upper bound: 0.7785256
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7723215
NS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7785256
NS_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7881663, upper bound: 0.7832194
NS_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7832194
NS_A1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7881663, upper bound: 0.7838832
NS_A1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7838832
NS_A1_B1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7766175
NS_A1_B1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7817862
NS_A1_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7723215
NS_A1_B1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7785256
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7862511
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7824318
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7862511
NS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7832750, upper bound: 0.7824318
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7814502, upper bound: 0.7714915
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7819102, upper bound: 0.7776602
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7714915
NS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7776602
NS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7853286
NS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7821339
NS_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7881660, upper bound: 0.7828662
NS_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7835745, upper bound: 0.7828662
NS_A1_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7750172
NS_A1_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7801889
NS_A1_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7714915
NS_A1_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7785256, upper bound: 0.7776602
NS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7862511, upper bound: 0.7832750
NS_A2_B1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7832750
NS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7862511, upper bound: 0.7839765
NS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7839765
NS_A2_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7814502
NS_A2_B1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7819102
NS_A2_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7786506
NS_A2_B1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7786506
NS_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7854322
NS_A2_B1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7822137
NS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7862511, upper bound: 0.7830123
NS_A2_B1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7824318, upper bound: 0.7830123
NS_A2_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7800391
NS_A2_B1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7802552
NS_A2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7853286, upper bound: 0.7835745
NS_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7835745
NS_A2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7881660
NS_A2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7835745
NS_A2_B2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7750172, upper bound: 0.7785256
NS_A2_B2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7801889, upper bound: 0.7785256
NS_A2_B2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7785256
NS_A2_B2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7785256
NS_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7862510
NS_A2_B2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7824318
NS_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7862510
NS_A2_B2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7821339, upper bound: 0.7824318
NS_A2_B2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7750172, upper bound: 0.7776602
NS_A2_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.81
Output dim: 0, lower bound: -0.7801889, upper bound: 0.7776602

## BFS NS instance: NS_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.5562359, 1.0441966, 0.3810282, 1.0689805, -0.5127445, 0.6631684
1: -0.1941883, 0.1892764, -0.2962954, 0.2889529, -0.4831412, 0.4855718
2: -0.1224670, 0.2709500, -0.2102395, 0.3875544, -0.5100214, 0.4811895
3: -0.1398863, 0.1998780, -0.2157398, 0.2908480, -0.4307343, 0.4156179
4: -0.2126476, 0.1675450, -0.3084385, 0.2713432, -0.4839908, 0.4759834
5: -0.2227700, 0.3031985, -0.3318299, 0.4342905, -0.6570605, 0.6350284
6: -0.1505518, 0.2117428, -0.2362154, 0.3211720, -0.4717239, 0.4479582
7: -0.2255598, 0.2160370, -0.3210292, 0.3294711, -0.5550309, 0.5370662
8: -0.2069509, 0.2648493, -0.3082706, 0.4001795, -0.6071304, 0.5731199
9: -0.1961929, 0.2590477, -0.3011312, 0.3876245, -0.5838174, 0.5601789

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7822967, upper bound: 0.7720928
time: 2.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7825723, upper bound: 0.7782530
time: 1.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.5676962, 1.0424891, 0.4123873, 1.0646309, -0.4969347, 0.6301017
1: -0.1875311, 0.1823938, -0.2780021, 0.2714691, -0.4590002, 0.4603959
2: -0.1172861, 0.2633453, -0.1940650, 0.3668148, -0.4841008, 0.4574103
3: -0.1346006, 0.1943357, -0.2024321, 0.2744167, -0.4090173, 0.3967679
4: -0.2062204, 0.1614037, -0.2915162, 0.2522275, -0.4584480, 0.4529199
5: -0.2149113, 0.2959654, -0.3129112, 0.4097670, -0.6246783, 0.6088766
6: -0.1459004, 0.2038576, -0.2202330, 0.3021892, -0.4480896, 0.4240907
7: -0.2195250, 0.2083058, -0.3037677, 0.3095094, -0.5290344, 0.5120735
8: -0.2001286, 0.2550976, -0.2903786, 0.3767037, -0.5768322, 0.5454762
9: -0.1892384, 0.2515122, -0.2825374, 0.3638859, -0.5531243, 0.5340496

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7720928
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7782530
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.4195517, 1.0636371, 0.4908091, 1.0537535, -0.6342018, 0.5728281
1: -0.2738228, 0.2674749, -0.2322551, 0.2277471, -0.5015700, 0.4997301
2: -0.1903698, 0.3620768, -0.1536170, 0.3149498, -0.5053197, 0.5156937
3: -0.1993918, 0.2706630, -0.1691527, 0.2333263, -0.4327181, 0.4398157
4: -0.2876503, 0.2478605, -0.2491978, 0.2044243, -0.4920745, 0.4970583
5: -0.3085891, 0.4041645, -0.2656003, 0.3484398, -0.6570289, 0.6697648
6: -0.2165818, 0.2978524, -0.1802651, 0.2547181, -0.4712999, 0.4781175
7: -0.2998243, 0.3049489, -0.2606015, 0.2595900, -0.5594144, 0.5655503
8: -0.2862911, 0.3713405, -0.2456355, 0.3179965, -0.6042876, 0.6169760
9: -0.2782895, 0.3584630, -0.2360391, 0.3045222, -0.5828117, 0.5945021

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7770086
time: 1.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7825309
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.4496972, 1.0594558, 0.5068898, 1.0515230, -0.6018258, 0.5525660
1: -0.2562376, 0.2506681, -0.2228745, 0.2187819, -0.4750195, 0.4735427
2: -0.1748215, 0.3421397, -0.1453229, 0.3043148, -0.4791363, 0.4874625
3: -0.1865992, 0.2548677, -0.1623287, 0.2249005, -0.4114996, 0.4171963
4: -0.2713830, 0.2294848, -0.2405203, 0.1946220, -0.4660050, 0.4700052
5: -0.2904027, 0.3805900, -0.2558989, 0.3358644, -0.6262671, 0.6364889
6: -0.2012180, 0.2796045, -0.1720696, 0.2449840, -0.4462020, 0.4516741
7: -0.2832311, 0.2857599, -0.2517502, 0.2493537, -0.5325848, 0.5375102
8: -0.2690918, 0.3487732, -0.2364607, 0.3059582, -0.5750500, 0.5852339
9: -0.2604156, 0.3356433, -0.2265045, 0.2923493, -0.5527649, 0.5621477

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7720928
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7782530
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.4559969, 1.0585819, 0.3783439, 1.0690706, -0.6130736, 0.6802381
1: -0.2525627, 0.2471558, -0.2968305, 0.2894124, -0.5419751, 0.5439863
2: -0.1715722, 0.3379731, -0.2107787, 0.3881055, -0.5596777, 0.5487518
3: -0.1839258, 0.2515667, -0.2164033, 0.2913524, -0.4752781, 0.4679700
4: -0.2679833, 0.2256446, -0.3089506, 0.2718954, -0.5398787, 0.5345952
5: -0.2866019, 0.3756636, -0.3328621, 0.4353005, -0.7219024, 0.7085258
6: -0.1980072, 0.2757910, -0.2365461, 0.3234731, -0.5214803, 0.5123371
7: -0.2797634, 0.2817496, -0.3214570, 0.3301038, -0.6098672, 0.6032066
8: -0.2654974, 0.3440571, -0.3091323, 0.4012094, -0.6667067, 0.6531894
9: -0.2566800, 0.3308743, -0.3015842, 0.3882285, -0.6449085, 0.6324586

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7822926, upper bound: 0.7720928
time: 2.06 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7825309, upper bound: 0.7782530
time: 1.90 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.4700984, 1.0566258, 0.4104328, 1.0647225, -0.5946242, 0.6461930
1: -0.2443366, 0.2392940, -0.2784874, 0.2718999, -0.5162365, 0.5177813
2: -0.1642990, 0.3286471, -0.1945362, 0.3673295, -0.5316285, 0.5231833
3: -0.1779416, 0.2441781, -0.2029653, 0.2748676, -0.4528092, 0.4471433
4: -0.2603740, 0.2170488, -0.2919760, 0.2527406, -0.5131146, 0.5090247
5: -0.2780946, 0.3646359, -0.3137268, 0.4106039, -0.6886986, 0.6783628
6: -0.1908204, 0.2672549, -0.2205700, 0.3037928, -0.4946132, 0.4878249
7: -0.2720015, 0.2727734, -0.3041767, 0.3100699, -0.5820714, 0.5769501
8: -0.2574520, 0.3335007, -0.2910806, 0.3775446, -0.6349965, 0.6245812
9: -0.2483190, 0.3201998, -0.2829729, 0.3644583, -0.6127773, 0.6031727

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7720928
time: 1.93 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7782530
time: 2.17 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.4559969, 1.0585819, 0.3426425, 1.0743048, -0.6183079, 0.7159395
1: -0.2525627, 0.2471558, -0.3186874, 0.3103536, -0.5629163, 0.5658432
2: -0.1715722, 0.3379731, -0.2300377, 0.4129413, -0.5845135, 0.5680109
3: -0.1839258, 0.2515667, -0.2320295, 0.3109608, -0.4948866, 0.4835962
4: -0.2679833, 0.2256446, -0.3291523, 0.2947418, -0.5627251, 0.5547969
5: -0.2866019, 0.3756636, -0.3549875, 0.4643087, -0.7509106, 0.7306511
6: -0.1980072, 0.2757910, -0.2557788, 0.3444080, -0.5424152, 0.5315698
7: -0.2797634, 0.2817496, -0.3421579, 0.3539055, -0.6336689, 0.6239076
8: -0.2654974, 0.3440571, -0.3301713, 0.4289153, -0.6944127, 0.6742284
9: -0.2566800, 0.3308743, -0.3238910, 0.4166818, -0.6733618, 0.6547653

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7770079, upper bound: 0.7785256
time: 1.89 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7825309, upper bound: 0.7785256
time: 1.83 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.4700984, 1.0566258, 0.3732384, 1.0700610, -0.5999626, 0.6833874
1: -0.2443366, 0.2392940, -0.3008395, 0.2932956, -0.5376322, 0.5401335
2: -0.1642990, 0.3286471, -0.2142571, 0.3927064, -0.5570055, 0.5429043
3: -0.1779416, 0.2441781, -0.2190455, 0.2949296, -0.4728712, 0.4632236
4: -0.2603740, 0.2170488, -0.3126420, 0.2760915, -0.5364654, 0.5296907
5: -0.2780946, 0.3646359, -0.3365294, 0.4403822, -0.7184768, 0.7011653
6: -0.1908204, 0.2672549, -0.2401854, 0.3258873, -0.5167077, 0.5074403
7: -0.2720015, 0.2727734, -0.3253168, 0.3344296, -0.6064311, 0.5980902
8: -0.2574520, 0.3335007, -0.3127150, 0.4060108, -0.6634628, 0.6462157
9: -0.2483190, 0.3201998, -0.3057498, 0.3935213, -0.6418403, 0.6259496

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7723215
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7785256
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.4513598, 1.0592251, 0.3582716, 1.0721370, -0.6207772, 0.7009535
1: -0.2552677, 0.2497412, -0.3095703, 0.3016400, -0.5569077, 0.5593115
2: -0.1739639, 0.3410399, -0.2219766, 0.4026051, -0.5765691, 0.5630165
3: -0.1858937, 0.2539964, -0.2253970, 0.3027718, -0.4886655, 0.4793934
4: -0.2704857, 0.2284714, -0.3207186, 0.2852147, -0.5557004, 0.5491900
5: -0.2893996, 0.3792900, -0.3455587, 0.4520864, -0.7414861, 0.7248486
6: -0.2003706, 0.2785980, -0.2478133, 0.3349473, -0.5353179, 0.5264114
7: -0.2823159, 0.2847014, -0.3335551, 0.3439568, -0.6262727, 0.6182565
8: -0.2681431, 0.3475285, -0.3212544, 0.4172154, -0.6853585, 0.6687828
9: -0.2594296, 0.3343846, -0.3146241, 0.4048506, -0.6642802, 0.6490088

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7754310
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7808210
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.4195517, 1.0636371, 0.3582716, 1.0721370, -0.6525853, 0.7053655
1: -0.2738228, 0.2674749, -0.3095703, 0.3016400, -0.5754628, 0.5770452
2: -0.1903698, 0.3620768, -0.2219766, 0.4026051, -0.5929750, 0.5840533
3: -0.1993918, 0.2706630, -0.2253970, 0.3027718, -0.5021636, 0.4960600
4: -0.2876503, 0.2478605, -0.3207186, 0.2852147, -0.5728649, 0.5685791
5: -0.3085891, 0.4041645, -0.3455587, 0.4520864, -0.7606755, 0.7497232
6: -0.2165818, 0.2978524, -0.2478133, 0.3349473, -0.5515292, 0.5456657
7: -0.2998243, 0.3049489, -0.3335551, 0.3439568, -0.6437811, 0.6385040
8: -0.2862911, 0.3713405, -0.3212544, 0.4172154, -0.7035066, 0.6925949
9: -0.2782895, 0.3584630, -0.3146241, 0.4048506, -0.6831401, 0.6730870

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7754310
time: 2.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7808210
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.3450854, 1.0739660, 0.3558425, 1.0722228, -0.7271374, 0.7181236
1: -0.3172624, 0.3089916, -0.3100705, 0.3020723, -0.6193347, 0.6190620
2: -0.2287777, 0.4113258, -0.2224785, 0.4031226, -0.6319004, 0.6338043
3: -0.2309927, 0.3096809, -0.2260007, 0.3032426, -0.5342353, 0.5356815
4: -0.3278340, 0.2932526, -0.3211966, 0.2857242, -0.6135582, 0.6144491
5: -0.3535138, 0.4623984, -0.3464962, 0.4530199, -0.8065337, 0.8088945
6: -0.2545337, 0.3429292, -0.2481287, 0.3370247, -0.5915584, 0.5910579
7: -0.3408131, 0.3523504, -0.3339587, 0.3445470, -0.6853601, 0.6863091
8: -0.3287777, 0.4270866, -0.3220364, 0.4181639, -0.7469416, 0.7491230
9: -0.3224426, 0.4148325, -0.3150521, 0.4054198, -0.7278624, 0.7298845

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7720928, upper bound: 0.7805282
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7808210
time: 2.18 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.3763446, 1.0696300, 0.3731122, 1.0688409, -0.6924963, 0.6965178
1: -0.2990274, 0.2915638, -0.2963912, 0.2888207, -0.5878481, 0.5879550
2: -0.2126550, 0.3906522, -0.2106194, 0.3874218, -0.6000769, 0.6012716
3: -0.2177274, 0.2933021, -0.2170521, 0.2910420, -0.5087693, 0.5103542
4: -0.3109658, 0.2741981, -0.3086029, 0.2714774, -0.5824431, 0.5828010
5: -0.3346555, 0.4379532, -0.3341087, 0.4357207, -0.7703762, 0.7720619
6: -0.2386023, 0.3240071, -0.2357021, 0.3288768, -0.5674791, 0.5597091
7: -0.3236070, 0.3324525, -0.3207842, 0.3297983, -0.6534052, 0.6532366
8: -0.3109429, 0.4036856, -0.3099427, 0.4018168, -0.7127597, 0.7136282
9: -0.3039083, 0.3911700, -0.3008341, 0.3873575, -0.6912657, 0.6920041

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7720928, upper bound: 0.7771760
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7771760
time: 2.42 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.4559969, 1.0585819, 0.2020860, 1.0958800, -0.6398831, 0.8564959
1: -0.2525627, 0.2471558, -0.4065972, 0.4030688, -0.6556315, 0.6537529
2: -0.1715722, 0.3379731, -0.3115996, 0.5120916, -0.6836638, 0.6495727
3: -0.1839258, 0.2515667, -0.2929218, 0.3903867, -0.5743125, 0.5444884
4: -0.2679833, 0.2256446, -0.4252234, 0.4170234, -0.6850067, 0.6508679
5: -0.2866019, 0.3756636, -0.4511495, 0.5725144, -0.8591163, 0.8268131
6: -0.1980072, 0.2757910, -0.3342881, 0.4481832, -0.6461904, 0.6100791
7: -0.2797634, 0.2817496, -0.4247312, 0.4487706, -0.7285340, 0.7064809
8: -0.2654974, 0.3440571, -0.4422578, 0.5398465, -0.8053439, 0.7863148
9: -0.2566800, 0.3308743, -0.4204756, 0.5205714, -0.7772514, 0.7513499

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7817147, upper bound: 0.7714915
time: 1.92 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7825309, upper bound: 0.7776602
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.4700984, 1.0566258, 0.2403663, 1.0900096, -0.6199112, 0.8162595
1: -0.2443366, 0.2392940, -0.3804635, 0.3753220, -0.6196586, 0.6197574
2: -0.1642990, 0.3286471, -0.2895969, 0.4848256, -0.6491247, 0.6182440
3: -0.1779416, 0.2441781, -0.2769375, 0.3640680, -0.5420096, 0.5211155
4: -0.2603740, 0.2170488, -0.3971440, 0.3776820, -0.6380560, 0.6141928
5: -0.2780946, 0.3646359, -0.4231251, 0.5431596, -0.8212543, 0.7877611
6: -0.1908204, 0.2672549, -0.3105594, 0.4192294, -0.6100498, 0.5778143
7: -0.2720015, 0.2727734, -0.4003727, 0.4229185, -0.6949201, 0.6731461
8: -0.2574520, 0.3335007, -0.4077620, 0.5094914, -0.7669433, 0.7412627
9: -0.2483190, 0.3201998, -0.3930619, 0.4926639, -0.7409829, 0.7132617

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7714915
time: 2.26 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7776602
time: 2.04 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.3582716, 1.0721370, 0.4513598, 1.0592251, -0.7009535, 0.6207772
1: -0.3095703, 0.3016400, -0.2552677, 0.2497412, -0.5593115, 0.5569077
2: -0.2219766, 0.4026051, -0.1739639, 0.3410399, -0.5630165, 0.5765691
3: -0.2253970, 0.3027718, -0.1858937, 0.2539964, -0.4793934, 0.4886655
4: -0.3207186, 0.2852147, -0.2704857, 0.2284714, -0.5491900, 0.5557004
5: -0.3455587, 0.4520864, -0.2893996, 0.3792900, -0.7248486, 0.7414861
6: -0.2478133, 0.3349473, -0.2003706, 0.2785980, -0.5264114, 0.5353179
7: -0.3335551, 0.3439568, -0.2823159, 0.2847014, -0.6182565, 0.6262727
8: -0.3212544, 0.4172154, -0.2681431, 0.3475285, -0.6687828, 0.6853585
9: -0.3146241, 0.4048506, -0.2594296, 0.3343846, -0.6490088, 0.6642802

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7754310, upper bound: 0.7782580
time: 1.70 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7808210, upper bound: 0.7782580
time: 1.57 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.3582716, 1.0721370, 0.4195517, 1.0636371, -0.7053655, 0.6525853
1: -0.3095703, 0.3016400, -0.2738228, 0.2674749, -0.5770452, 0.5754628
2: -0.2219766, 0.4026051, -0.1903698, 0.3620768, -0.5840533, 0.5929750
3: -0.2253970, 0.3027718, -0.1993918, 0.2706630, -0.4960600, 0.5021636
4: -0.3207186, 0.2852147, -0.2876503, 0.2478605, -0.5685791, 0.5728649
5: -0.3455587, 0.4520864, -0.3085891, 0.4041645, -0.7497232, 0.7606755
6: -0.2478133, 0.3349473, -0.2165818, 0.2978524, -0.5456657, 0.5515292
7: -0.3335551, 0.3439568, -0.2998243, 0.3049489, -0.6385040, 0.6437811
8: -0.3212544, 0.4172154, -0.2862911, 0.3713405, -0.6925949, 0.7035066
9: -0.3146241, 0.4048506, -0.2782895, 0.3584630, -0.6730870, 0.6831401

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7754310, upper bound: 0.7786506
time: 1.92 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7808210, upper bound: 0.7786506
time: 1.99 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.3762134, 1.0687426, 0.4496972, 1.0594558, -0.6832423, 0.6190455
1: -0.2957953, 0.2883111, -0.2562376, 0.2506681, -0.5464634, 0.5445487
2: -0.2100142, 0.3868108, -0.1748215, 0.3421397, -0.5521539, 0.5616323
3: -0.2162848, 0.2904783, -0.1865992, 0.2548677, -0.4711525, 0.4770775
4: -0.3080314, 0.2708446, -0.2713830, 0.2294848, -0.5375162, 0.5422276
5: -0.3329061, 0.4345768, -0.2904027, 0.3805900, -0.7134961, 0.7249795
6: -0.2353409, 0.3262148, -0.2012180, 0.2796045, -0.5149454, 0.5274328
7: -0.3203114, 0.3290896, -0.2832311, 0.2857599, -0.6060713, 0.6123207
8: -0.3089343, 0.4006491, -0.2690918, 0.3487732, -0.6577076, 0.6697409
9: -0.3003340, 0.3866890, -0.2604156, 0.3356433, -0.6359773, 0.6471046

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7709765, upper bound: 0.7786506
time: 1.91 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7786506
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.2425796, 1.0893581, 0.4286729, 1.0623717, -0.8197922, 0.6606852
1: -0.3780392, 0.3726310, -0.2685020, 0.2623896, -0.6404288, 0.6411330
2: -0.2873366, 0.4820393, -0.1856654, 0.3560441, -0.6433808, 0.6677047
3: -0.2754628, 0.3620429, -0.1955212, 0.2658836, -0.5413464, 0.5575641
4: -0.3944121, 0.3733480, -0.2827280, 0.2423004, -0.6367126, 0.6560761
5: -0.4207699, 0.5404676, -0.3030863, 0.3970315, -0.8178014, 0.8435539
6: -0.3079351, 0.4178966, -0.2119330, 0.2923311, -0.6002662, 0.6298295
7: -0.3978750, 0.4203667, -0.2948036, 0.2991428, -0.6970177, 0.7151703
8: -0.4044373, 0.5066215, -0.2810870, 0.3645121, -0.7689495, 0.7877085
9: -0.3901654, 0.4896750, -0.2728812, 0.3515581, -0.7417235, 0.7625562

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7709765, upper bound: 0.7805235
time: 1.81 seconds

## Relational analysis of NS_A2_B1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7808402
time: 1.70 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.3582716, 1.0721370, 0.2889470, 1.0824758, -0.7242042, 0.7831900
1: -0.3095703, 0.3016400, -0.3511562, 0.3428452, -0.6524155, 0.6527962
2: -0.2219766, 0.4026051, -0.2605705, 0.4501067, -0.6720833, 0.6631756
3: -0.2253970, 0.3027718, -0.2554035, 0.3387596, -0.5641565, 0.5581753
4: -0.3207186, 0.2852147, -0.3629163, 0.3325011, -0.6532197, 0.6481310
5: -0.3455587, 0.4520864, -0.3899683, 0.5057890, -0.8513477, 0.8420547
6: -0.2478133, 0.3349473, -0.2828213, 0.3819000, -0.6297133, 0.6177686
7: -0.3335551, 0.3439568, -0.3717121, 0.3898664, -0.7234215, 0.7156689
8: -0.3212544, 0.4172154, -0.3670068, 0.4707648, -0.7920192, 0.7842222
9: -0.3146241, 0.4048506, -0.3586593, 0.4567480, -0.7713721, 0.7635099

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7754310, upper bound: 0.7777442
time: 1.68 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7808210, upper bound: 0.7777442
time: 1.90 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.3558425, 1.0722228, 0.3450854, 1.0739660, -0.7181236, 0.7271374
1: -0.3100705, 0.3020723, -0.3172624, 0.3089916, -0.6190620, 0.6193347
2: -0.2224785, 0.4031226, -0.2287777, 0.4113258, -0.6338043, 0.6319004
3: -0.2260007, 0.3032426, -0.2309927, 0.3096809, -0.5356815, 0.5342353
4: -0.3211966, 0.2857242, -0.3278340, 0.2932526, -0.6144491, 0.6135582
5: -0.3464962, 0.4530199, -0.3535138, 0.4623984, -0.8088945, 0.8065337
6: -0.2481287, 0.3370247, -0.2545337, 0.3429292, -0.5910579, 0.5915584
7: -0.3339587, 0.3445470, -0.3408131, 0.3523504, -0.6863091, 0.6853601
8: -0.3220364, 0.4181639, -0.3287777, 0.4270866, -0.7491230, 0.7469416
9: -0.3150521, 0.4054198, -0.3224426, 0.4148325, -0.7298845, 0.7278624

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7805282, upper bound: 0.7720928
time: 1.71 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7808210, upper bound: 0.7782530
time: 2.38 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.3731122, 1.0688409, 0.3763446, 1.0696300, -0.6965178, 0.6924963
1: -0.2963912, 0.2888207, -0.2990274, 0.2915638, -0.5879550, 0.5878481
2: -0.2106194, 0.3874218, -0.2126550, 0.3906522, -0.6012716, 0.6000769
3: -0.2170521, 0.2910420, -0.2177274, 0.2933021, -0.5103542, 0.5087693
4: -0.3086029, 0.2714774, -0.3109658, 0.2741981, -0.5828010, 0.5824431
5: -0.3341087, 0.4357207, -0.3346555, 0.4379532, -0.7720619, 0.7703762
6: -0.2357021, 0.3288768, -0.2386023, 0.3240071, -0.5597091, 0.5674791
7: -0.3207842, 0.3297983, -0.3236070, 0.3324525, -0.6532366, 0.6534052
8: -0.3099427, 0.4018168, -0.3109429, 0.4036856, -0.7136282, 0.7127597
9: -0.3008341, 0.3873575, -0.3039083, 0.3911700, -0.6920041, 0.6912657

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7720928
time: 2.34 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7782530
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.2020860, 1.0958800, 0.4559969, 1.0585819, -0.8564959, 0.6398831
1: -0.4065972, 0.4030688, -0.2525627, 0.2471558, -0.6537529, 0.6556315
2: -0.3115996, 0.5120916, -0.1715722, 0.3379731, -0.6495727, 0.6836638
3: -0.2929218, 0.3903867, -0.1839258, 0.2515667, -0.5444884, 0.5743125
4: -0.4252234, 0.4170234, -0.2679833, 0.2256446, -0.6508679, 0.6850067
5: -0.4511495, 0.5725144, -0.2866019, 0.3756636, -0.8268131, 0.8591163
6: -0.3342881, 0.4481832, -0.1980072, 0.2757910, -0.6100791, 0.6461904
7: -0.4247312, 0.4487706, -0.2797634, 0.2817496, -0.7064809, 0.7285340
8: -0.4422578, 0.5398465, -0.2654974, 0.3440571, -0.7863148, 0.8053439
9: -0.4204756, 0.5205714, -0.2566800, 0.3308743, -0.7513499, 0.7772514

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7817147
time: 1.51 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7825309
time: 2.44 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.2403663, 1.0900096, 0.4700984, 1.0566258, -0.8162595, 0.6199112
1: -0.3804635, 0.3753220, -0.2443366, 0.2392940, -0.6197574, 0.6196586
2: -0.2895969, 0.4848256, -0.1642990, 0.3286471, -0.6182440, 0.6491247
3: -0.2769375, 0.3640680, -0.1779416, 0.2441781, -0.5211155, 0.5420096
4: -0.3971440, 0.3776820, -0.2603740, 0.2170488, -0.6141928, 0.6380560
5: -0.4231251, 0.5431596, -0.2780946, 0.3646359, -0.7877611, 0.8212543
6: -0.3105594, 0.4192294, -0.1908204, 0.2672549, -0.5778143, 0.6100498
7: -0.4003727, 0.4229185, -0.2720015, 0.2727734, -0.6731461, 0.6949201
8: -0.4077620, 0.5094914, -0.2574520, 0.3335007, -0.7412627, 0.7669433
9: -0.3930619, 0.4926639, -0.2483190, 0.3201998, -0.7132617, 0.7409829

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7782530
time: 1.38 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7782530
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.2397356, 1.0894573, 0.3243972, 1.0769894, -0.8372538, 0.7650601
1: -0.3785933, 0.3731779, -0.3295698, 0.3205257, -0.6991190, 0.7027477
2: -0.2879410, 0.4826305, -0.2394482, 0.4250876, -0.7130286, 0.7220787
3: -0.2761283, 0.3625638, -0.2397721, 0.3206693, -0.5967976, 0.6023358
4: -0.3950387, 0.3741408, -0.3389979, 0.3060276, -0.7010663, 0.7131387
5: -0.4218369, 0.5415114, -0.3659945, 0.4786893, -0.9005262, 0.9075060
6: -0.3083424, 0.4205120, -0.2651548, 0.3554523, -0.6637947, 0.6856667
7: -0.3983532, 0.4210410, -0.3522007, 0.3658540, -0.7642071, 0.7732417
8: -0.4054458, 0.5077188, -0.3410261, 0.4425738, -0.8480196, 0.8487449
9: -0.3907088, 0.4902700, -0.3347602, 0.4304931, -0.8212019, 0.8250301

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7709765, upper bound: 0.7805182
time: 1.78 seconds

## Relational analysis of NS_A2_B2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7808210
time: 1.58 seconds

## BFS NS instance: NS_A2_B2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.2020860, 1.0958800, 0.3243972, 1.0769894, -0.8749034, 0.7714828
1: -0.4065972, 0.4030688, -0.3295698, 0.3205257, -0.7271229, 0.7326386
2: -0.3115996, 0.5120916, -0.2394482, 0.4250876, -0.7366872, 0.7515398
3: -0.2929218, 0.3903867, -0.2397721, 0.3206693, -0.6135911, 0.6301588
4: -0.4252234, 0.4170234, -0.3389979, 0.3060276, -0.7312510, 0.7560213
5: -0.4511495, 0.5725144, -0.3659945, 0.4786893, -0.9298388, 0.9385090
6: -0.3342881, 0.4481832, -0.2651548, 0.3554523, -0.6897404, 0.7133380
7: -0.4247312, 0.4487706, -0.3522007, 0.3658540, -0.7905852, 0.8009713
8: -0.4422578, 0.5398465, -0.3410261, 0.4425738, -0.8848315, 0.8808726
9: -0.4204756, 0.5205714, -0.3347602, 0.4304931, -0.8509687, 0.8553315

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_B2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7754310
time: 1.70 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_B2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7808210
time: 1.74 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.24 seconds
NS_A1_B1_A1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7822967, upper bound: 0.7720928
NS_A1_B1_A1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7825723, upper bound: 0.7782530
NS_A1_B1_A1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7720928
NS_A1_B1_A1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7782530
NS_A1_B1_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7770086
NS_A1_B1_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7825309
NS_A1_B1_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7720928
NS_A1_B1_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7782530
NS_A1_B1_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7822926, upper bound: 0.7720928
NS_A1_B1_A2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7825309, upper bound: 0.7782530
NS_A1_B1_A2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7720928
NS_A1_B1_A2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7782530
NS_A1_B1_A2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7770079, upper bound: 0.7785256
NS_A1_B1_A2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7825309, upper bound: 0.7785256
NS_A1_B1_A2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7723215
NS_A1_B1_A2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7785256
NS_A1_B2_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7754310
NS_A1_B2_A1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7808210
NS_A1_B2_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7754310
NS_A1_B2_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7808210
NS_A1_B2_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7720928, upper bound: 0.7805282
NS_A1_B2_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7808210
NS_A1_B2_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7720928, upper bound: 0.7771760
NS_A1_B2_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7771760
NS_A1_B2_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7817147, upper bound: 0.7714915
NS_A1_B2_A2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7825309, upper bound: 0.7776602
NS_A1_B2_A2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7714915
NS_A1_B2_A2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7782530, upper bound: 0.7776602
NS_A2_B1_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7754310, upper bound: 0.7782580
NS_A2_B1_B1_A1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7808210, upper bound: 0.7782580
NS_A2_B1_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7754310, upper bound: 0.7786506
NS_A2_B1_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7808210, upper bound: 0.7786506
NS_A2_B1_B1_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7709765, upper bound: 0.7786506
NS_A2_B1_B1_A1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7786506
NS_A2_B1_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7709765, upper bound: 0.7805235
NS_A2_B1_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7808402
NS_A2_B1_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7754310, upper bound: 0.7777442
NS_A2_B1_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7808210, upper bound: 0.7777442
NS_A2_B2_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7805282, upper bound: 0.7720928
NS_A2_B2_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7808210, upper bound: 0.7782530
NS_A2_B2_B1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7720928
NS_A2_B2_B1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7782530
NS_A2_B2_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7817147
NS_A2_B2_B1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7825309
NS_A2_B2_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7714915, upper bound: 0.7782530
NS_A2_B2_B1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7782530
NS_A2_B2_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7709765, upper bound: 0.7805182
NS_A2_B2_B2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7771760, upper bound: 0.7808210
NS_A2_B2_B2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7754310
NS_A2_B2_B2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.24
Output dim: 0, lower bound: -0.7776602, upper bound: 0.7808210

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.44 + 501.73 = 507.17 seconds
